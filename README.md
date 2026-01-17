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
