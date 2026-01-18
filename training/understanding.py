import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from accelerate import Accelerator
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import model  # 
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
# --- 1. Dataset Definition ---
class EmoUnderstandingDataset(Dataset):
    def __init__(self, data_root, size=(224, 224), processor=processor):
        self.data_root = data_root
        self.image_paths = []
        self.emotions = []
        self.emotion_to_idx = {
            "amusement": 0, "awe": 1, "contentment": 2, "excitement": 3,
            "disgust": 4, "fear": 5, "sadness": 6, "anger": 7
        }
        for emo_name in self.emotion_to_idx.keys():
            dir_path = os.path.join(data_root, emo_name)
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png'))]
                for file in files:
                    self.image_paths.append(os.path.join(dir_path, file))
                    self.emotions.append(emo_name)

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.processor = processor
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        path = self.image_paths[i]
        emo_name = self.emotions[i]
        image = Image.open(path).convert("RGB")
        # inputs = self.processor(images=image, text="",return_tensors="pt")
        # pixel_values = inputs["pixel_values"].squeeze(0)
        return {
            "image": image,
            "emotion_name": emo_name,
            "label_idx": torch.tensor(self.emotion_to_idx[emo_name])
        }

# --- 2. Utility Functions ---
def format_qwen_prompt(question):
    # 
    return (
        f"<|im_start|>system\nYou are a helpful assistant. You must answer with ONLY ONE WORD from the given list.<|im_end|>\n"
        f"<|im_start|>user\nClassify the emotion: amusement, awe, contentment, excitement, disgust, fear, sadness, or anger.<|im_end|>\n"
        f"<|im_start|>assistant\nPlease answer: Yes/No.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

import re
def collate_fn(examples):
    return {
        "image": [ex["image"] for ex in examples], # 收集为 PIL 列表
        "emotion_name": [ex["emotion_name"] for ex in examples],
        "label_idx": torch.stack([ex["label_idx"] for ex in examples])
    }
# def evaluate_accuracy(trainer, test_loader, device):
#     trainer.grpo.eval()
#     correct = 0
#     total = 0
#     emotions = ["amusement", "awe", "contentment", "excitement", "disgust", "fear", "sadness", "anger"]

#     with torch.no_grad():
#         for batch in test_loader:
#             image_tensor = batch["image"].to(device, dtype=torch.bfloat16)
            
#             # 1. 
#             gt_emotion = batch["emotion_name"]
#             if isinstance(gt_emotion, (list, tuple)):
#                 gt_emotion = gt_emotion[0]
#             gt_emotion = str(gt_emotion).lower().strip()

#             # 2.
#             eval_prompts = [
#                 f"Is this image {gt_emotion}? Answer strictly with 'Yes' or 'No'.",
#                 "Classify the emotion. Answer ONLY one word: amusement, awe, contentment, excitement, disgust, fear, sadness, or anger.",
#                 "Score 0-10 for misleading intensity. Answer with a single number only."
#             ]

#             # 3. 
#             _, _, generated_texts = trainer.generate_multiple_answers(image_tensor, eval_prompts)
#             full_prediction = generated_texts[0] # 
            
#             # 4. 
#             ans2_match = re.search(r"Answer 2[:：]\s*(.*?)(?=Answer 3|$)", full_prediction, re.DOTALL | re.IGNORECASE)
            
#             if ans2_match:
#                 # 
#                 pred_content = ans2_match.group(1).lower().strip()
                
#                 # 
#                 if pred_content.startswith(gt_emotion) or gt_emotion in pred_content:
#                     pred_label = gt_emotion
#                 else:
#                     # 
#                     pred_label = "wrong_emotion"
#                     for emo in emotions:
#                         if emo in pred_content:
#                             pred_label = emo
#                             break
#             else:
#                 pred_label = "format_error"

#             # 5. 
#             is_match = (pred_label == gt_emotion)

#             if total < 3: # 
#                 print(f"\n[Eval Debug Task-Combined]")
#                 print(f"GT Emotion: {gt_emotion}")
#                 print(f"Raw Output: {full_prediction[:150]}...") # 
#                 print(f"Extracted Answer 2: '{pred_label}' | Match: {is_match}")

#             if is_match:
#                 correct += 1
#             total += 1
            
#     trainer.grpo.train() # 
#     return (correct / total * 100) if total > 0 else 0
class EmotionEvaluator:
    """
    
    """
    def __init__(self, categories=None):
        self.categories = categories or [
            "amusement", "awe", "contentment", "excitement", 
            "disgust", "fear", "sadness", "anger"
        ]
        # 
        self.boundary_pattern = r'\b{emotion}\b'
        self.exclusion_keywords = ["not", "no", "than", ","]

    def is_correct(self, full_prediction, gt_emotion, pred_multi=""):
        full_prediction = full_prediction.lower()
        gt_emotion = gt_emotion.lower()
        pred_multi = pred_multi.lower()

        # 
        if gt_emotion in pred_multi:
            return True

        # 
        header_part = full_prediction.split("answer 2")[0]
        if "yes" in header_part and gt_emotion in full_prediction:
            return True

        # 
        # pattern = self.boundary_pattern.format(emotion=re.escape(gt_emotion))
        # if re.search(pattern, full_prediction):
        #     #
        #     if not any(f"{key} {gt_emotion}" in full_prediction for key in self.exclusion_keywords):
        #         #
        #         if f", {gt_emotion}" not in full_prediction:
        #             return True
        
        return False

def evaluate_accuracy(trainer, test_loader, device,accelerator):
    trainer.grpo.eval()
    evaluator = EmotionEvaluator() # 
    correct, total = 0, 0

    # 
    PROMPT_TEMPLATES = [
        "Is this image {gt}? Answer strictly with 'Yes' or 'No'.",
        "What emotion is expressed in this image? Choose from: {cats}.",
        "Please provide a reasoning for your answer, focusing on the objects and scene.",
        "Score 0-10 for your belief that this image represents OTHER emotions than {gt}."
    ]

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"]
            gt_emotion = str(batch["emotion_name"][0] if isinstance(batch["emotion_name"], (list, tuple)) else batch["emotion_name"]).lower().strip()

            # 
            current_prompts = [
                p.format(gt=gt_emotion, cats=", ".join(evaluator.categories)) for p in PROMPT_TEMPLATES
            ]

            _, _, generated_texts = trainer.generate_multiple_answers(images, current_prompts)
            full_prediction = generated_texts[0]

            # 
            ans2_match = re.search(r"Answer 2[:：]\s*([^\n]+)", full_prediction, re.I)
            pred_multi = ans2_match.group(1).strip() if ans2_match else ""

            # 
            if evaluator.is_correct(full_prediction, gt_emotion, pred_multi):
                correct += 1
            total += 1

            if total <= 3:
                if accelerator.is_main_process:
                   print(f"[Debug] Target: {gt_emotion} | Match: {evaluator.is_correct(full_prediction, gt_emotion, pred_multi)}")

    trainer.grpo.train()
    accuracy = (correct / total * 100) if total > 0 else 0
    if accelerator.is_main_process:
      print(f"\n>> Final Eval Accuracy: {accuracy:.2f}%")
    return accuracy
def run_test(trainer, test_loader, device, step):
    trainer.grpo.eval()
    batch = next(iter(test_loader))
    # image_tensor = batch["image"].to(device, dtype=torch.bfloat16)
    images = batch["image"]
    emo_name = batch["emotion_name"][0]
    test_prompts = [
        f"Is this image {emo_name}? Answer strictly with 'Yes' or 'No'.",
        "What emotion is expressed in this image? Choose from: amusement, awe, contentment, excitement, disgust, fear, sadness, anger.",
        "Please provide a reasoning for your answer, especially paying attention to the object and scene of this image.",
        f"Score 0-10 for your belief that this image represents OTHER emotions than {emo_name}. Answer with a single number only."
    ]
    
    with torch.no_grad():
       
        _, _, generated_texts = trainer.generate_multiple_answers(images, test_prompts)
        
        print(f"\n[Visual Test at Step {step}]")
        print(f"Ground Truth: {emo_name}")
        
        print(f"Model Full Output:\n{generated_texts[0].strip()}")
    
    trainer.grpo.train()

# --- 3. Main Training Function ---
def train():
    accelerator = Accelerator(gradient_accumulation_steps=4)
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 2
    device = accelerator.device
    # target_dtype = torch.bfloat16
    trainer = model.UnderstandingTraining(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        num_answers=2,
        kl_coef=0.1,
        quantize_4bit=True,
        device = device,
        accelerator=accelerator
    )
    # --- 在这之前完成 trainer 的初始化 ---
    
    target_dtype = torch.bfloat16 # 确保这与你的 deepspeed 配置一致
    
    # 1. 暴力转换所有可训练参数
    if accelerator.is_main_process:
     print("Force aligning all trainable parameters to target dtype...")
    for name, param in trainer.named_parameters():
        if param.requires_grad:
            if param.dtype != target_dtype:
                # print(f"Converting {name} from {param.dtype} to {target_dtype}")
                param.data = param.data.to(target_dtype)
    
    # 2. 检查是否有遗漏的 Dtypes
    active_dtypes = {p.dtype for p in trainer.parameters() if p.requires_grad}
    if accelerator.is_main_process:
      print(f"Trainable parameter dtypes now: {active_dtypes}")
    
    if len(active_dtypes) > 1:
        raise ValueError(f"Still found mixed dtypes: {active_dtypes}. DeepSpeed ZeRO-3 will fail.")

    # 3. 重新获取参数列表给优化器
    # 必须确保优化器里拿到的参数引用已经是转换后的
    
    full_dataset = EmoUnderstandingDataset(data_root="/root/EmoGen/EmoSet-118K/image")
    all_indices = np.arange(len(full_dataset))
    np.random.shuffle(all_indices)

    # 
    train_subset_indices = all_indices[:2560]
    test_subset_indices = all_indices[2330:2585] # 
    
    train_loader = DataLoader(Subset(full_dataset, train_subset_indices), batch_size=2, shuffle=True,collate_fn=collate_fn)
    
    test_loader = DataLoader(Subset(full_dataset, test_subset_indices), batch_size=2, shuffle=False,collate_fn=collate_fn)

    trainable_params = [p for p in trainer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)

    # 4. 执行 prepare
    trainer, optimizer, train_loader, test_loader = accelerator.prepare(
        trainer, optimizer, train_loader, test_loader
    )
    num_epochs = 20
    global_step = 0
    best_acc = 0.0
    output_dir = "./checkpoints_grpo"
    os.makedirs(output_dir, exist_ok=True)
    if accelerator.is_main_process:
      print(f"Starting training on {device}... Train Size: {len(train_subset_indices)}, Test Size: {len(test_subset_indices)}")

    for epoch in range(num_epochs):
        trainer.grpo.train()
        epoch_rewards = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(trainer.grpo):
                images = batch["image"]
                emo_name = batch["emotion_name"][0]
                label_idx = batch["label_idx"].to(device)

                labels = {
                    "binary_label": torch.tensor([1]).to(device), 
                    "multi_label": label_idx,
                    "misleading_label": torch.tensor([0]).to(device) 
                }

                # prompts = [
                #     format_qwen_prompt(f"Is this image {emo_name}? Answer strictly with 'Yes' or 'No'."),
                #     format_qwen_prompt("Classify the emotion. Answer ONLY one word from the list: amusement, awe, contentment, excitement, disgust, fear, sadness, anger."),
                #     format_qwen_prompt(f"Score 0-10 for misleading intensity regarding {emo_name}. Answer with a single number only.")
                # ]
                
                prompts = [
               f"Is this image {emo_name}? Answer strictly with 'Yes' or 'No'.",
        "What emotion is expressed in this image? Choose from: amusement, awe, contentment, excitement, disgust, fear, sadness, anger.",
        "Please provide a reasoning for your answer, especially paying attention to the object and scene of this image.",
        f"Score 0-10 for your belief that this image represents OTHER emotions than {emo_name}. Answer with a single number only."
            ]
                outputs = trainer.forward(
                    images=images,
                    prompts=prompts,
                    labels=labels,
                    emotion_name=emo_name,
                    return_loss=True
                )

                loss = outputs["loss"]
                reward_mean = outputs["reward_mean"]
                del outputs
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                epoch_rewards.append(reward_mean.item())
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "rew": f"{reward_mean.item():.3f}"})

                # 
                if global_step % 1000 == 0 and accelerator.is_main_process:
                    # 1. save Checkpoint
                    trainer.save_ckpt(output_dir, global_step)
                    
                    # 2. visualization
                    run_test(trainer, test_loader, device, global_step)
                    
                    # 3. evaluate accuracy
                    accuracy = evaluate_accuracy(trainer, test_loader, device,accelerator)
                    if accelerator.is_main_process:
                     print(f"\n>>> Global Step {global_step} | Test Accuracy: {accuracy:.2f}%")
                    
                    # 4. save the best model
                    if accuracy > best_acc:
                        best_acc = accuracy
                        best_path = os.path.join(output_dir, "best_model")
                        os.makedirs(best_path, exist_ok=True)
                        # save Lora and Head
                        trainer.model.save_pretrained(os.path.join(best_path, "lora_best"))

                        torch.save(trainer.emotion_head.state_dict(), os.path.join(best_path, "head_best.pth"))
                        if accelerator.is_main_process:
                         print(f"!!! New Best Accuracy! Model saved to {best_path}")
                    if accelerator.is_main_process:
                     print("-" * 50)

        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        if accelerator.is_main_process:
         print(f"\n[Epoch {epoch} Finished] Average Reward: {avg_reward:.4f}\n")

if __name__ == "__main__":
    train()