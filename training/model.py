"""
Model definitions for EmoGen-IVT-LR modification.
This file is structured to be a drop-in replacement for EmoGen/model.py.

Origins and modifications:
- The original EmoGen uses an image encoder -> emotion space -> mapper to CLIP token embeddings.
- We replace this with IVT-LR-style latent reasoning guided MLLM components.
- We keep the original MLP/FC mapper classes for backward compatibility, but the new default is IVTLRReasoner.

Important: No direct imports from external repos are used. IVT-LR ideas are adapted
into a self-contained Reasoner that interleaves vision features and text prompts,
and learns a small aggregation head to produce token embeddings that condition diffusion.
"""
import os
import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, PeftModel, TaskType  # LoRA
import bitsandbytes as bnb  # 
from torch.optim import AdamW
# import bitsandbytes as bnb  # 
from sentence_transformers import SentenceTransformer, util  
# 在 model.py 顶部
from accelerate import Accelerator
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
import re
class BackBone(nn.Module):
    """
    Minimal image backbone placeholder to match EmoGen dependency.
    In the original repo, BackBone is defined elsewhere. Here we expose a basic ResNet50 feature extractor
    while keeping the interface similar. This is used only if you explicitly choose FC/MLP mappers.
    """
    def __init__(self):
        super().__init__()
        try:
            import torchvision.models as models
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # remove the classifier head, keep avgpool output as feature
            modules = list(resnet.children())[:-1]
            self.encoder = nn.Sequential(*modules)
            self.out_dim = 2048
        except Exception:
            # fallback tiny convnet
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.out_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: Bx3xHxW in [-1,1] normalized
        feat = self.encoder(x)  # BxCx1x1
        feat = feat.view(feat.size(0), -1)  # BxC
        return feat


class FC(nn.Module):
    def __init__(self, in_dim: int = 2048, embed_dim: int = 768):
        super().__init__()
        self.fc = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, num_layers: int = 2, need_ReLU: bool = True, need_LN: bool = False, need_Dropout: bool = False,
                 in_dim: int = 2048, embed_dim: int = 768):
        super().__init__()
        layers = []
        dim = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(dim, dim))
            if need_LN:
                layers.append(nn.LayerNorm(dim))
            if need_ReLU:
                layers.append(nn.ReLU(inplace=True))
            if need_Dropout:
                layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(dim, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleMLP(nn.Module):
    def __init__(self, need_ReLU: bool = True, need_Dropout: bool = False,
                 in_dim: int = 2048, embed_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True) if need_ReLU else nn.Identity(),
            nn.Dropout(0.1) if need_Dropout else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class emo_classifier(nn.Module):
    """
    Placeholder emotion classifier interface to match original training.
    In a typical EmoGen environment, you have a pre-trained classifier loaded from weights/Clip_emotion_classifier/...
    Here we provide a tiny classifier head that expects CLIP pooled embeddings (dimension 768).
    """
    def __init__(self, embed_dim: int = 768, num_classes: int = 8):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class IVTLRReasoner(nn.Module):
    """
    IVT-LR style latent reasoning guided MLLM.

    Design:
    - Vision encoder: CLIP ViT-L/14 to extract image features (pooled).
    - Language generator: FLAN-T5-small to produce latent step-by-step reasoning given emotion + attribute.
    - Fusion: We interleave image features and token embeddings by generating a short reasoning text,
      then encode this text with CLIP text projection. A small trainable adapter aggregates
      (vision_feat, text_feat) to produce the final token embedding for the placeholder token(s).

    Note: We keep everything lightweight and self-contained. Large MLLMs can be swapped
    by editing the model_name in init. The trainable part is the adapter; encoders can be frozen
    by setting freeze_backbones=True.
    """
    def __init__(self, clip_model_name: str = "model/clip-vit-large-patch14",
                 lm_name: str = "google/flan-t5-small",
                 num_vectors: int = 1,
                 embed_dim: int = 768,
                 freeze_backbones: bool = True):
        super().__init__()
        # Vision-Text backbone
        try:
            self.clip = CLIPModel.from_pretrained(clip_model_name)
            self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        except Exception:
            # Fallback to HF hub path if local path not available
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_embed_dim = self.clip.text_projection.out_features if hasattr(self.clip, "text_projection") else embed_dim

        # Language model for latent reasoning text
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.lm = AutoModelForSeq2SeqLM.from_pretrained(lm_name)

        # Adapter to map fused representation to token embedding(s)
        self.num_vectors = num_vectors
        self.embed_dim = embed_dim
        self.adapter = nn.Sequential(
            nn.Linear(self.clip_embed_dim * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_vectors * embed_dim),
        )

        if freeze_backbones:
            for p in self.clip.parameters():
                p.requires_grad = False
            for p in self.lm.parameters():
                p.requires_grad = False

    def generate_reasoning(self, emotion: str, attribute: str) -> str:
        prompt = (
            f"You are a latent multimodal reasoner. Given an intended emotion '{emotion}' and attribute '{attribute}', "
            f"produce a concise conceptual description and key emotional cues for diffusion guidance."
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.adapter[0].weight.device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = self.lm.generate(**inputs, max_length=64, num_beams=2)
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return text

    def encode_text_clip(self, text: str, device: torch.device) -> torch.Tensor:
        data = self.processor(text=[text], images=None, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            clip_out = self.clip.get_text_features(**data)
        # map to CLIP embedding space using text_projection if available
        if hasattr(self.clip, "text_model") and hasattr(self.clip, "text_projection"):
            ids = self.processor.tokenizer(text, return_tensors="pt", padding=True).to(device).input_ids
            tm_out = self.clip.text_model(ids)
            pooled = tm_out[1]
            proj = self.clip.text_projection(pooled)
            return proj
        return clip_out

    def encode_image_clip(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # image_tensor: Bx3x224x224 in [0,1] normalized to CLIP stats? We expect preprocessed.
        # Convert tensor to PIL via processor if needed; however we can directly forward using CLIP's vision_model
        device = image_tensor.device
        with torch.no_grad():
            # CLIP expects pixel_values from its processor
            # We convert back to PIL and re-process for robustness
            imgs = []
            for img in image_tensor:
                # de-normalize from [-1,1] to [0,1]
                arr = img.detach().cpu().clamp(-1, 1)
                arr = (arr + 1.0) / 2.0
                arr = (arr * 255.0).byte().permute(1, 2, 0).numpy()
                from PIL import Image
                imgs.append(Image.fromarray(arr))
            data = self.processor(images=imgs, return_tensors="pt").to(device)
            clip_out = self.clip.get_image_features(**data)
        return clip_out

    def forward(self, image_feat_tensor: torch.Tensor, emotions: List[str], attributes: List[str]) -> torch.Tensor:
        """
        Produce num_vectors x embed_dim tensor to replace placeholder token embedding(s).
        - image_feat_tensor: batch of images processed as in TextualInversionDataset.tfm (Bx3x224x224, normalized)
        - emotions/attributes: lists of strings for each element in batch. We use the first element for guidance
          to produce a single set of token vectors shared across batch to match original training design.
        """
        device = image_feat_tensor.device
        # Encode image via CLIP
        img_clip = self.encode_image_clip(image_feat_tensor)  # BxClipDim
        img_vec = img_clip.mean(dim=0, keepdim=True)  # 1xClipDim
        # Generate reasoning text and encode with CLIP text projection
        emo = emotions[0] if len(emotions) > 0 else ""
        attr = attributes[0] if len(attributes) > 0 else ""
        text = self.generate_reasoning(emo, attr)
        txt_vec = self.encode_text_clip(text, device) # 1xClipDim
        # Fuse
        fused = torch.cat([img_vec, txt_vec], dim=-1)  # 1x(2*ClipDim)
        out = self.adapter(fused)  # 1x(num_vectors*embed_dim)
        out = out.view(self.num_vectors, self.embed_dim)
        return out


def load_reasoner(args_embed_dim: int = 768, num_vectors: int = 1) -> IVTLRReasoner:
    """Factory to construct the IVTLRReasoner with common defaults."""
    reasoner = IVTLRReasoner(num_vectors=num_vectors, embed_dim=args_embed_dim)
    return reasoner
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)

# import re
import torch
# from sentence_transformers import SentenceTransformer, util  


class PromptTemplateManager:
    """
   Prompt template: Define the formats of the inputs and the outputs
    """
    def __init__(self, device: torch.device):
        self.device = device
        # Define the Prompt Templates：key is the question type，value includes the input template、output_format、expected_answer
        self.prompt_templates = {
            "Binary_choice": {
                "input_template": "Does the emotion of the following image belong to <target_emotion> <image>\\n Requirement: Only output 'Yes' or 'No' (without quotation marks)",
                "output_format": r"^(Yes|No)$",
                "expected_answer": None  
            },
            "Multiple_choice": {
                "input_template": "Please choose which emotion the following image belong to:<image>\\n Requirement:Only output one of the followings:amusement、awe、contentment、excitement、disgust、fear、sadness、anger",
                "output_format": r"^[amusement|awe|contentment|excitement|disgust|fear|sadness|anger]$",
                "expected_answer": None
            },
            "Misleading_choice": {
                "input_template": "Rate to what extent you believe that the following image belong to <misleading_emotion>, from 0-100. <image>\\nRequirement:Give a number between 0 and 100",
                "output_format": r"^[0-9]{1,3}$",  
                "expected_answer": None
            },
            "output_contract": {
                "thought_tag": "THOUGHT",
                "answer_tag": "ANSWER",
                "answer_format": r"^(Yes|No)$"
                
            },
        }
        # Similarity model: to match the answers that don't align with any template
        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)

    def format_prompt(self, question_type: str, **kwargs) -> str:
        """
        Replace the place holder: e.g: <****>
        :param question_type: question type, corresponding to the question type
        :param kwargs: place holder parameters,including image_path、emotion_label
        :return: Formalized prompt strings
        """
        if question_type not in self.prompt_templates:
            raise ValueError(f"Unkown question types: {question_type}")
        template = self.prompt_templates[question_type]["input_template"]
        # replace the place holder
        for key, value in kwargs.items():
            template = template.replace(f"<{key}>", str(value))
        return template

    def check_output_format(self, question_type: str, model_output: str) -> bool:
        """
        check whether the model output matches the expected format
        :param question_type
        :param model_output
        :return: whether the format matches
        """
        template = self.prompt_templates[question_type]
        if "output_contract" in template:
            contract = template["output_contract"]

            parsed = self.parse_structured_output(
                model_output,
                thought_tag=contract.get("thought_tag", "THOUGHT"),
                answer_tag=contract.get("answer_tag", "ANSWER")
            )

            if not parsed["valid_structure"]:
              return False
            
            answer_re = contract["answer_format"]
            return bool(re.match(answer_re, parsed["answer"]))
        format_re = template["output_format"]
        return bool(re.match(format_re, model_output.strip()))

    def match_answer(self, question_type: str, model_output: str, expected_answer: str or list, match_type: str = "exact", thought_min_len: int = 10) -> float:
        """
        match the model output with the expected answer
        :param question_type
        :param model_output
        :param expected_answer
        :param match_type: keyword、semantic、exact
        :return: （0-1）
        """
        template = self.prompt_templates[question_type]
        if "output_contract" in template:
            contract = template["output_contract"]
            parsed = self.parse_structured_output(
                model_output,
                contract["thought_tag"],
                contract["answer_tag"]
            )
            if not parsed["valid_structure"]:
              return 0.0
            
            thought_score = 1.0
            if parsed["thought"] is None or len(parsed["thought"]) < thought_min_len:
                thought_score = 0.5  
            answer = parsed["answer"]
            if match_type == "exact":
                answer_score = 1.0 if answer == expected_answer else 0.0
            elif match_type == "semantic":
                emb1 = self.sim_model.encode(answer, convert_to_tensor=True, device=self.device)
                emb2 = self.sim_model.encode(expected_answer, convert_to_tensor=True, device=self.device)
                answer_score = util.cos_sim(emb1, emb2).item()
            else:
               raise ValueError("Unsupported match_type")
            return thought_score * answer_score
       
        if match_type == "exact":
            return 1.0 if model_output.strip() == expected_answer else 0.0
        elif match_type == "semantic":
            emb1 = self.sim_model.encode(model_output, convert_to_tensor=True, device=self.device)
            emb2 = self.sim_model.encode(expected_answer, convert_to_tensor=True, device=self.device)
            return util.cos_sim(emb1, emb2).item()
        else:
            raise ValueError("Unsupported match_type")
        
    def parse_structured_output(
            self,
            model_output: str,
            thought_tag: str = "THOUGHT",
            answer_tag: str = "ANSWER"
        ) -> dict:
         """
            Parse structured output like:
            <THOUGHT>...</THOUGHT><ANSWER>...</ANSWER>
        """
         result = {
        "thought": None,
        "answer": None,
        "valid_structure": False
         }
         thought_pattern = rf"<{thought_tag}>(.*?)</{thought_tag}>"
         answer_pattern = rf"<{answer_tag}>(.*?)</{answer_tag}>"
         thought_match = re.search(thought_pattern, model_output, re.S)
         answer_match = re.search(answer_pattern, model_output, re.S)
         if thought_match and answer_match:
            result["thought"] = thought_match.group(1).strip()
            result["answer"] = answer_match.group(1).strip()
            result["valid_structure"] = True

         return result
class UnderstandingTraining(nn.Module):
    """Training class for the understanding model.
        Step I: Train on the EmoSet
        Step II: Train on the filtered EmoSet 
        Support GRPO relative advantage emotion understanding model training class:
        1. Generate G answers per sample → 2. Compute multi-task Reward → 3. GRPO group normalization → 4. KL constraint + gradient update
    """
    def __init__( self,
        model_name: str = "Qwen/Qwen2.5-VL-2B-Instruct",  
        num_emotion_classes: int = 8,                     
        lora_rank: int = 64,                               
        lora_alpha: int = 16,                            
        lora_dropout: float = 0.05,                       
        grpo_alpha: float = 0.5,                         
        grpo_beta: float = 0.1, 
        kl_coef: float = 0.1,                          
        quantize_4bit: bool = True,                       
        freeze_vision_tower: bool = True,                 
        learning_rate: float = 1e-4,                      
        warmup_steps: int = 100,                          
        total_training_steps: int = 10000, 
        num_answers: int = 4,  # number of answers G per sample              
        device: Optional[torch.device] = None,
        accelerator = None ):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_emotion_classes = num_emotion_classes
        self.total_training_steps = total_training_steps
        self.num_answers = num_answers
        #min_pixels = 256 * 28 * 28  
        max_pixels = 256 * 28 * 28
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_name, trust_remote_code=True, max_pixels=max_pixels)
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     model_name,
        #     trust_remote_code = True,
        #     padding_side = "right",
        #     use_fast = True
        # )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        bnb_config = None
        if quantize_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        # accelerator = Accelerator()
        self.device = device
        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            # device_map="auto" if self.device.type == "cuda" else None,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32
        )
        self.base_model.to(device)

        if freeze_vision_tower and hasattr(self.base_model, "vision_tower"):
            for param in self.base_model.vision_tower.parameters():
                param.requires_grad = False
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  
            bias="none",
            inference_mode=False
        )

        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters() 
        self.grpo = GRPOGradientReweight(
            num_emotion_classes=num_emotion_classes,
            alpha=grpo_alpha,
            beta=grpo_beta,
            kl_coef=kl_coef
        ).to(device=self.device, dtype=torch.bfloat16)

        self.emotion_head = nn.Linear(
            self.base_model.config.hidden_size,
            num_emotion_classes
        ).to(device=self.device, dtype=torch.bfloat16)
        # self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     model_name,
        #     quantization_config=bnb_config,
        #     device_map="auto" if self.device.type == "cuda" else None,
        #     trust_remote_code=True,
        #     torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32
        # )
        self.optimizer = self._setup_optimizer(learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )
        self.model.gradient_checkpointing_enable()

        self.model.base_model.gradient_checkpointing_enable()
        self.accelerator = accelerator
    def _setup_optimizer(self, lr: float) -> AdamW:
        trainable_params = list(self.model.parameters()) + list(self.emotion_head.parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": self.emotion_head.parameters(),
                "lr": lr * 10,  
                "weight_decay": 0.01
            }
        ]
        return AdamW(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.999))
    
    def generate_multiple_answers(self, images: torch.Tensor, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multiple answers per sample for GRPO training
        :param images: image tensor,with shape (B, C, H, W)
        :param prompts: prompt list for each sample, in the shape of (B,)
        :return: model_logits (B*G, C), hidden_states (B*G, D)
        """
        if isinstance(images, list):
          batch_size = len(images)
        else:
          batch_size = images.shape[0]
        num_answers = self.num_answers
        num_tasks = len(prompts)

        # expand the input, for each sample, repeat G times
        flat_images = []
        for img in images:
          flat_images.extend([img] * num_answers)
        # extended_prompts = []
        # for p in prompts:
        #     extended_prompts.extend([p] * num_answers)
        combined_instruction = (
        f"Please observe the image and answer the following four questions at once:\n"
        f"1. {prompts[0]}\n"
        f"2. {prompts[1]}\n"
        f"3. {prompts[2]}\n\n"
        f"4. {prompts[3]}\n\n"
        "Format your response as:\n"
        "Answer 1: ...\nAnswer 2: ...\nAnswer 3: ...\nAnswer 4: ..."
    )
        formatted_prompt = (
        f"<|im_start|>system\nYou are a helpful assistant,capable of reasoning the emotion of the given images,especially paying attention to the object and scene of the image.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{combined_instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
        extended_prompts = [formatted_prompt] * num_answers
        flat_prompts = [formatted_prompt] * (batch_size * num_answers)
        # Tokenize input（Qwen 2.5 VL multimodal input）
        
        inputs = self.processor(
            text=flat_prompts,
            images=flat_images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # propogate through the model
        outputs = self.model(**inputs, output_hidden_states=True)
        model_logits = self.emotion_head(outputs.hidden_states[-1][:, 0, :])  # (B*G, C)
        hidden_states = outputs.hidden_states[-1][:, 0, :]  # (B*G, D)

        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=100,  
            num_beams=1, 
            do_sample=True,     
            temperature=0.7     
        )
        input_len = inputs.input_ids.shape[1]
        actual_generated_ids = generated_ids[:, input_len:]
        generated_texts = self.tokenizer.batch_decode(actual_generated_ids, skip_special_tokens=True)
        
        return model_logits, hidden_states, generated_texts
    # 
   

    def compute_text_reward(self, generated_texts, gt_emotion, labels):
        """
    
        generated_texts: List[str], 长度为 G (4)
        """
        num_answers = self.num_answers # 4
        rewards = torch.zeros(num_answers).to(self.device)
        emotions = ["amusement", "awe", "contentment", "excitement", "disgust", "fear", "sadness", "anger"]

        for i in range(num_answers):
            full_text = generated_texts[i]
            score = 0.0
            
            # 
            ans1_match = re.search(r"Answer 1[:：]\s*(.*?)(?=Answer 2|$)", full_text, re.DOTALL | re.IGNORECASE)
            ans2_match = re.search(r"Answer 2[:：]\s*(.*?)(?=Answer 3|$)", full_text, re.DOTALL | re.IGNORECASE)
            ans3_match = re.search(r"Answer 3[:：]\s*(.*?)(?=Answer 4|$)", full_text, re.DOTALL | re.IGNORECASE)
            ans4_match = re.search(r"Answer 4[:：]\s*(.*)", full_text, re.DOTALL | re.IGNORECASE)
            # 
            t1 = ans1_match.group(1).lower().strip() if ans1_match else ""
            t2 = ans2_match.group(1).lower().strip() if ans2_match else ""
            t3 = ans3_match.group(1).lower().strip() if ans3_match else ""
            t4 = ans4_match.group(1).lower().strip() if ans4_match else ""
            # 
            if ans1_match and ans2_match and ans3_match and ans4_match:
                score += 1.0
            else:
                score -= 1.0

            # 
            gt_yes_no = "yes" if labels["binary_label"] == 1 else "no"
            if t1.startswith(gt_yes_no):
                score += 1.0  #
            elif gt_yes_no in t1:
                score += 1.0
            else:
                score -= 1.0

            # -
            if t2.startswith(gt_emotion.lower()):
                score += 1.0  # 
            elif gt_emotion.lower() in t2:
                score += 1.0 
            else:
                score -= 1.0 
            
            if gt_emotion.lower() in t3 and ("object" in t3 or "scene" in t3):
                score += 2.0
            elif gt_emotion.lower() in t3:
                score += 1.0
            else:
                score -= 1.0
          
            
            nums = re.findall(r'\d+', t4)
            if nums:
                val = int(nums[0])
                target_val = labels["misleading_label"].item()
                # if val == target_val:
                #     score += 2.0
                # elif 0 <= val <= 10:
                #     score += 0.5
                if val >= 0 and val <= 10:
                    score += (10 - val) / 10.0
                else:
                    score -= 1.0
            rewards[i] = score

    
        if rewards.std() > 1e-4:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        else:
            #
            if rewards.max() > 2.0:
                 pass
            else:
                rewards += torch.randn_like(rewards) * 0.01

        # 修改前：
# rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)


        # if rewards.max() < 2.0: 
        #   rewards = rewards - 5.0 
        # else:
        #   rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        # return rewards
        # --- 最终推荐的归一化逻辑 ---
        # max_reward = rewards.max().item()
        
        # if max_reward < 2.0: 
        #     # 情况 A: 全组都烂，集体惩罚
        #     rewards = rewards - 5.0 
        # else:
        #     std = rewards.std()
        #     if std > 1e-4:
        #         # 情况 B: 组内有差异，执行标准归一化（拉开好坏差距）
        #         rewards = (rewards - rewards.mean()) / (std + 1e-6)
        #     else:
               
        #         pass
        return rewards
    # def forward(
    #     self,
    #     images: torch.Tensor,
    #     prompts: List[str],
    #     labels: Dict[str, torch.Tensor],
    #     emotion_name: str,
    #     return_loss: bool = True
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     Propogate：Multi answer generation → Reward Computation → GRPO Reweight
    #     :param images: Image Tensor Format，(B, C, H, W)
    #     :param prompts: Prompt list for each sample, in the shape of (B,)
    #     :param labels: label dictionary, including binary_label/multi_label/misleading_label
    #     :param return_loss: whether to return the loss for training
    #     :return: （logits/rewards/loss）
    #     """
    #     batch_size = images.shape[0]
    #     num_answers = self.num_answers

    #     # 1. Generate multiple answers per sample
    #     model_logits, hidden_states, generated_texts = self.generate_multiple_answers(images, prompts)
    #     num_prompts = len(prompts)
    #     if images.shape[0] == 1 and num_prompts > 1:
    #         expanded_images = images.repeat(num_prompts, 1, 1, 1)
    #     else:
    #         expanded_images = images
    #     # 2. Reference model logits
    #     ref_inputs = self.processor(
    #         text = prompts,
    #         images=expanded_images,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True
    #     ).to(self.device)
    #     ref_outputs = self.ref_model(**ref_inputs, output_hidden_states=True)
    #     ref_logits = self.emotion_head(ref_outputs.hidden_states[-1][:, 0, :])  # (B, C)

    #     # 3. Compute multi-task Reward
    #     gt_emo = emotion_name
    #     rewards = self.compute_text_reward(generated_texts, gt_emo, labels)

    #     result = {
    #         "model_logits": model_logits,
    #         "ref_logits": ref_logits,
    #         "rewards": rewards,
    #         "reward_mean": rewards.mean()
    #     }

    #     # 4. GRPO gradient reweighting (during training)
    #     if return_loss:
    #         # Collect trainable parameters
    #         trainable_params = list(self.model.parameters()) + list(self.emotion_head.parameters())

    #         # Compute the log probability of the selected answers
    #         log_probs = F.log_softmax(model_logits, dim=-1).gather(
    #             dim=-1,
    #             index=labels["multi_label"].repeat_interleave(num_answers).unsqueeze(-1)
    #         ).squeeze()

    #         # GRPO gradient reweighting
    #         total_loss = self.grpo.reweight_gradient(
    #             model_params=trainable_params,
    #             logits=model_logits,
    #             ref_logits=ref_logits,
    #             rewards=rewards,
    #             emotion_labels=labels["multi_label"],
    #             num_answers=num_answers,
    #             batch_size = len(prompts),
    #             log_probs=log_probs
    #         )
    #         #total_loss.backward()
    #         # Update class counts for GRPO
    #         self.grpo.update_class_counts(labels["multi_label"])

    #         result["loss"] = total_loss
    #         result["log_probs"] = log_probs

    #     return result
    def forward(
        self,
        images: torch.Tensor,
        prompts: List[str],
        labels: Dict[str, torch.Tensor],
        emotion_name: str,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        if isinstance(images, list):
         batch_size = len(images)
        else:
         batch_size = images.shape[0]
        num_answers = self.num_answers
        num_prompts = len(prompts)

        # --- ---
        # 
        with torch.no_grad():
            model_logits, hidden_states, generated_texts = self.generate_multiple_answers(images, prompts)
            
            #
            gt_emo = emotion_name
            rewards = self.compute_text_reward(generated_texts, gt_emo, labels)

        result = {
            "model_logits": model_logits,
            "rewards": rewards,
            "reward_mean": rewards.mean()
        }
        if self.accelerator.is_main_process:
         print("return_loss:", return_loss)
       
        if return_loss:
            # 
            torch.cuda.empty_cache() 
            needs_gold_reference = True
            for i in range(num_answers):
                text_to_check = generated_texts[i] # 
                ans1 = re.search(r"Answer 1[:：]\s*([^\n]+)", text_to_check, re.I)
                ans2 = re.search(r"Answer 2[:：]\s*([^\n]+)", text_to_check, re.I)

                # 2. 
                str_q1 = ans1.group(1).strip().strip('.,。，') if ans1 else ""
                str_q2 = ans2.group(1).strip().strip('.,。，') if ans2 else ""
                is_sample_correct = (str_q1.lower() == "yes") and (str_q2.lower() == emotion_name.lower())
                if is_sample_correct:
                    needs_gold_reference = False
                    break
            if self.accelerator.is_main_process:
             print("Needs gold reference:", needs_gold_reference)

            all_log_probs = []
            full_texts = []
            # for i in range(num_prompts):
            #     for g in range(num_answers):
            #         full_texts.append(prompts[i] + generated_texts[i * num_answers + g])
            single_combined_prompt = (
                "Please observe the image and answer the following three questions at once, strictly adhering to the output format:\n\n"
                f"Question 1: {prompts[0]}\n"
                f"Question 2: {prompts[1]}\n"
                f"Question 3:{prompts[2]}\n"
                f"Question 3: {prompts[3]}\n\n"
                "The response format must be:\n"
                "Answer 1: [Answer here]\n"
                "Answer 2: [Answer here]\n"
                "Answer 3: [Answer here]\n"
                "Answer 4: [Answer here]"
            )
            # 
            formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{single_combined_prompt}<|im_end|>\n<|im_start|>assistant\n"

            for g in range(num_answers):
                # 
                full_texts.append(formatted_prompt + generated_texts[g])
            # 
            if needs_gold_reference:
                with torch.no_grad():
                    with self.model.disable_adapter():
                        # teacher_prompt = (
                        #     f"The emotion of this image is {emotion_name}. "
                        #     f"Please provide a professional reasoning for this based on the image's objects and scene. "
                        #     f"Keep it concise."
                        # )
                        teacher_prompt = (
                            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                            f"The emotion of this image is {emotion_name}. "
                            f"Please find any subtle visual cues (lighting, shadows, composition, or hidden expressions) that support why it is {emotion_name}. Be creative and persuasive. Mention the {emotion_name} in your answer for many times. "
                            f"Keep it concise.<|im_end|>\n<|im_start|>assistant\n"
                        )
                       
                        t_in = self.processor(text=teacher_prompt, images=[images[0]], return_tensors="pt").to(self.device)
                        t_ids = self.model.generate(**t_in, max_new_tokens=100)
                        teacher_reasoning = self.processor.batch_decode(t_ids[:, t_in.input_ids.shape[1]:], skip_special_tokens=True)
                gt_text = (
                        f"Answer 1: Yes\n"
                        f"Answer 2: {emotion_name}\n"
                        f"Answer 3: {teacher_reasoning}\n"
                        f"Answer 4: 0"
                    )
                full_texts.append(formatted_prompt + gt_text)
                gold_reward = torch.tensor([1.5], device=self.device)
                rewards_for_loss = torch.cat([rewards, gold_reward])
                num_answers += 1
            else:
                rewards_for_loss = rewards
            
            micro_batch_size = 1 
            for i in range(0, len(full_texts), micro_batch_size):
                sub_texts = full_texts[i : i + micro_batch_size]
                sub_inputs = self.processor(
                    text=sub_texts, 
                    images=[images[0]] * len(sub_texts), 
                    return_tensors="pt", 
                    padding=True
                ).to(self.device)

              
                sub_outputs = self.model(**sub_inputs)
                
                shift_logits = sub_outputs.logits[..., :-1, :].contiguous()
                shift_labels = sub_inputs.input_ids[..., 1:].contiguous()
                
                sub_lp = F.log_softmax(shift_logits, dim=-1).gather(
                    dim=-1, index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)
                
                all_log_probs.append(sub_lp.mean(dim=-1))

            log_probs = torch.cat(all_log_probs, dim=0) # (B*G,)

            # Reference model log probs
            all_ref_lp = []
            with torch.no_grad():
                for i in range(0, len(full_texts), micro_batch_size):
                    sub_texts = full_texts[i : i + micro_batch_size]
                    sub_inputs = self.processor(text=sub_texts, images=[images[0]]*len(sub_texts), return_tensors="pt", padding=True).to(self.device)
                    # ref_outputs = self.ref_model(**sub_inputs)
                    with torch.no_grad():
                        with self.model.disable_adapter(): # 关键：不增加显存占用，直接调用底座
                           ref_outputs = self.model(**sub_inputs)
                    
                    shift_ref_logits = ref_outputs.logits[..., :-1, :].contiguous()
                    shift_labels = sub_inputs.input_ids[..., 1:].contiguous()
                    r_lp = F.log_softmax(shift_ref_logits, dim=-1).gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
                    all_ref_lp.append(r_lp.mean(dim=-1))
            
            ref_log_probs = torch.cat(all_ref_lp, dim=0)

            # Collect trainable parameters
            trainable_params = list(self.model.parameters()) + list(self.emotion_head.parameters())
            target_label = labels["multi_label"][0].unsqueeze(0)
            # log_probs = F.log_softmax(model_logits, dim=-1).gather(
            #     dim=-1,
            #     index=target_label.repeat_interleave(num_answers).unsqueeze(-1)
            # ).squeeze()
            total_loss = self.grpo.reweight_gradient(
                model_params=trainable_params,
                logits=log_probs,          
                ref_logits=ref_log_probs,  
                rewards=rewards_for_loss,           
                emotion_labels=target_label,
                num_answers=num_answers,
                batch_size=1,
                log_probs=log_probs        
            )

        # --- 打印逻辑修正 ---
        if self.accelerator.is_main_process:
         print(f"\nThe real emotion label is: {emotion_name}")
        
        for i in range(num_answers):
            full_text = full_texts[i]
            if "assistant\n" in full_text:
                full_text = full_text.split("assistant\n")[-1]
            else:
                full_text = full_text
            # 使用正则拆分出三个回答，用于打印显示
            ans1 = re.search(r"Answer 1[:：]\s*(.*?)(?=Answer 2|$)", full_text, re.S | re.I)
            ans2 = re.search(r"Answer 2[:：]\s*(.*?)(?=Answer 3|$)", full_text, re.S | re.I)
            ans3 = re.search(r"Answer 3[:：]\s*(.*?)(?=Answer 4|$)", full_text, re.S | re.I)
            ans4 = re.search(r"Answer 4[:：]\s*(.*)", full_text, re.S | re.I)

            # 提取内容，如果没有匹配到则显示原句的一部分（防止模型格式错误）
            str_q1 = ans1.group(1).strip() if ans1 else "N/A (Format Error)"
            str_q2 = ans2.group(1).strip() if ans2 else "N/A (Format Error)"
            str_q3 = ans3.group(1).strip() if ans3 else "N/A (Format Error)"
            str_q4 = ans4.group(1).strip() if ans4 else "N/A (Format Error)"
            if self.accelerator.is_main_process:
             print("-" * 30)
            if i < 2:
                if self.accelerator.is_main_process:
                 print(f"Sample {i+1} Total Reward: {rewards[i].item():.4f}")
            else:
                if self.accelerator.is_main_process:
                 print(f"Gold Reference Total Reward: {rewards_for_loss[i].item():.4f}")
            if self.accelerator.is_main_process:
                print("Corresponds to (Extracted):")
                print(f"  [Q1 - Binary]:     {str_q1}")
                print(f"  [Q2 - Multiple]:   {str_q2}")
                print(f"  [Q3 - Reasoning]: {str_q3}")
                print(f"  [Q4 - Misleading]: {str_q4}")
                # 如果你想看模型输出的原始全貌，可以取消下面这行的注释
                #print(f"  [Raw Output]: {full_text.replace('\n', ' ')}") 
                print("-" * 30)

        
        result["loss"] = total_loss
        result["log_probs"] = log_probs
        del all_log_probs, all_ref_lp, sub_outputs, ref_outputs
        return result
        return result
    def save_ckpt(self, save_path: str, step: int):
        """Save the model checkpoint"""
        os.makedirs(save_path, exist_ok=True)
        # Save the LoRA weights
        self.model.save_pretrained(os.path.join(save_path, f"lora_ckpt_step_{step}"))
        # Save the emotion classification head
        torch.save(
            self.emotion_head.state_dict(),
            os.path.join(save_path, f"emotion_head_step_{step}.pth")
        )
        # Save Tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))

    def load_ckpt(self, ckpt_path: str, step: int):
        """Load the model checkpoint"""
        # Load the LoRA weights
        accelerator = Accelerator()
        device = accelerator.device
        self.model = PeftModel.from_pretrained(
            self.base_model,
            os.path.join(ckpt_path, f"lora_ckpt_step_{step}"),
            # device_map="auto" if self.device.type == "cuda" else None
        )
        self.model.to(device)
        # Load the emotion classification head
        self.emotion_head.load_state_dict(
            torch.load(os.path.join(ckpt_path, f"emotion_head_step_{step}.pth"), map_location=self.device)
        )



import torch.nn.functional as F
class GRPOGradientReweight(nn.Module):
    """
    GRPO Module
    """
    def __init__(self, num_emotion_classes: int = 8, alpha: float = 0.5, beta: float = 0.1, kl_coef: float = 0.1):
        super().__init__()
        self.num_classes = num_emotion_classes
        self.alpha = alpha  # category balancing alpha
        self.beta = beta    # 
        self.kl_coef = kl_coef  # KL divergence coefficient =
        self.class_counts = nn.Parameter(torch.ones(num_emotion_classes), requires_grad=False)
        self.eps = 1e-8  # numerical stability

    def update_class_counts(self, emotion_labels: torch.Tensor):
        """update class counts based on the emotion labels in the current batch"""
        unique_labels, counts = torch.unique(emotion_labels, return_counts=True)
        for lbl, cnt in zip(unique_labels, counts):
            self.class_counts[lbl] += cnt

    def compute_group_advantage(self, rewards: torch.Tensor, batch_size: int, num_answers: int) -> torch.Tensor:
        """
        Compute group advantage：
        Normalize rewards within each group of answers for each sample.
        :param rewards: The rewards tensor of shape (B*G,)
        :param batch_size: Batch size B
        :param num_answers: Number of answers per sample G
        :return: Advantage tensor of shape (B*G,)
        """
        # Reshape rewards to (B, G)
        rewards_reshaped = rewards.reshape(batch_size, num_answers)
        
        # 1. Compute group mean and std
        group_mean = rewards_reshaped.mean(dim=1, keepdim=True)  # (B, 1)
        group_std = rewards_reshaped.std(dim=1, keepdim=True)    # (B, 1)
        
        # 2. Compute advantage within each group
        advantage_reshaped = (rewards_reshaped - group_mean) / (group_std + self.eps)
        
        # 3. Reshape back to (B*G,), matching the original reward shape
        advantage = advantage_reshaped.reshape(-1)
        
        return advantage
    
    def compute_advantage(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """compute advantage as rewards - values"""
        advantage = rewards - values
        return advantage

    def compute_kl_loss(self, model_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss between model and reference probabilities.
        :param model_logits: Logits from the model (B*G, C)
        :param ref_logits: Logits from the reference model (B, C)
        :return: KL divergence loss (scalar)
        """
        model_probs = F.softmax(model_logits, dim=-1)
        ref_probs = F.softmax(ref_logits.detach(), dim=-1) 
        kl_loss = F.kl_div(model_probs.log(), ref_probs, reduction="batchmean")
        return kl_loss

    # def reweight_gradient(
    #     self, 
    #     model_params: List[nn.Parameter], 
    #     logits: torch.Tensor, 
    #     ref_logits: torch.Tensor,
    #     rewards: torch.Tensor, 
    #     emotion_labels: torch.Tensor,
    #     num_answers: int,
    #     batch_size: int,
    #     log_probs: torch.Tensor
    # ) -> torch.Tensor:
    #     """
        
    #     :param model_logits: The output of the yet-to-optimize model logits (B*G, C)
    #     :param ref_logits: The reference model logits (B, C)
    #     :param rewards: total Reward (B*G,)
    #     :param emotion_labels: emotion label (B,)
    #     :param num_answers: number of answers per sample G
    #     :param log_probs: log probabilities of the selected answers (B*G,)
    #     :return: weighted total loss
    #     """
    #     # 1. compute Advantage
    #     advantage = self.compute_group_advantage(rewards, batch_size, num_answers)
    #     #advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # 标准化

    #     # 2. compute class weights
    #     class_freq = self.class_counts / self.class_counts.sum()
    #     class_weight = 1.0 / (class_freq[emotion_labels] + 1e-6)
    #     class_weight = class_weight / class_weight.mean()
    #     # Expand to G answers (each sample's G answers share class weights)
    #     class_weight = class_weight.repeat_interleave(logits.shape[0] // class_weight.shape[0])

    #     # 3. compute policy gradient loss: log_prob * advantage * class_weight
    #     log_probs = F.log_softmax(logits, dim=-1).gather(dim=-1, index=emotion_labels.repeat_interleave(logits.shape[0] // emotion_labels.shape[0]).unsqueeze(-1)).squeeze()
    #     policy_loss = -(log_probs * advantage * class_weight).mean()
        
        
    #     expanded_ref_logits = ref_logits.repeat_interleave(num_answers, dim=0)
    #     # 4. compute KL divergence loss
    #     kl_loss = self.compute_kl_loss(logits, expanded_ref_logits)

    #     # 5. total loss
    #     total_loss = policy_loss + self.kl_coef * kl_loss
    #     # total_loss.backward()  # backpropagate

    #     return total_loss
    def reweight_gradient(self, logits, ref_logits, rewards, num_answers, batch_size, log_probs, **kwargs):
        """Reweight gradients using GRPO strategy."""
        rewards_group = rewards.view(batch_size, num_answers)
        mean = rewards_group.mean(dim=1, keepdim=True)
        std = rewards_group.std(dim=1, keepdim=True) + 1e-4
        advantages = ((rewards_group - mean) / std).view(-1) # 回到 [12]

       
        policy_loss = -(advantages.detach() * log_probs).mean()

       
        kl_loss = torch.pow(log_probs - ref_logits.detach(), 2).mean()

        total_loss = policy_loss + self.kl_coef * kl_loss
        return total_loss