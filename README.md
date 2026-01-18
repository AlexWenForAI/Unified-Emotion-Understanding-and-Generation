# Unified-Emotion-Understanding-and-Generation
# This repository is for a unified emotion understanding and generation framework, aiming to enhance the emotion valence and fidelity in the image understanding and generation process.
> [Boxuan Wen](https://boxuanwen.github.io/),[Yijie Zhu](https://scholar.google.com/citations?user=0GtAUPoAAAAJ&hl=en), [Zitong Yu](https://sites.google.com/view/zitongyu)
> Great Bay Area University, KTH Royal Institute of Technology

## Description
Implementation of the unified architecture, in particular GRPO traning pipelines of the understanding part.

## Pipeline
<p align="left">
<img src="docs/Understanding.png" width="1200px"/>  
<br>
Fig 1. Finetuning of the understanding model, emphasizing on enhance the model's understanding in the following aspects: 1. Binary choice: tell whether the image belongs to a specific emotion category or not. 2. Multiple Choice: tell which emotion category the image belongs to, among the 8 given choices. 3.Misleading choice: given a easily confused option, asking the model to decide whether it's correct or not. 
</p>

## Setup
To create the conda environment needed to run the code, run the following command:

```
conda env create -f environment/env.yaml
conda activate py39
```
Alternatively, install the requirements from `requirements.txt`

## Usage
### Preliminary
[EmoSet](https://vcc.tech/EmoSet) is needed to train in this network, as attribute label is necessary.

We need to organize the dataset according to its attributes, and the following is its layout:
```
data_root
|
├── object
|    ├── (3) cart
|    |    ├── disgust_05770.jpg
|    |    ├── ...
|    |    └── sadness_10803.jpg
|    ├── ...
|    └── (13094) plant
|
└── scene
     ├── (1) airfield
     ├── ...
     └── (2406) street
```
The number before the attribute represents the total number of images with this attribute.
### Training
To train our network, follow these steps:

First, manually modify the code related to reading EmoSet and change the file location to the location where your EmoSet is located. For example:
In training/dataset_balance.py
```
annotion_path = f'/mnt/d/dataset/EmoSet/annotation/{emotion}/{emotion}_{number}.json' # change to "{your_EmoSet_location}/annotation/{emotion}/{emotion}_{number}.json"
```

Secondly, create training dataset:
```
python training/dataset_balance.py
```

Thirdly, start to train your own network:
```
accelerate training/main.py
```

Additionally, generate emotional image:
```
python training/inference.py
```
You can modify config/config.yaml to change some details.

Finally, training the understanding module:
```
python understanding.py
```
In order to accelerate the training, you can execuete:
```
nohup accelerate launch --use_deepspeed --zero_stage [number of GPUs you wish to use] understanding.py > training_output_0107.log 2>&1 &
```

