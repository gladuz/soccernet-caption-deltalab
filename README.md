# SoccerNet 2024 - Dense Video Captioning

## Introduction
This repository contains the code and models for our SoccerNet 2024 Dense Video Captioning submission from [DeLTA Lab](https://sites.google.com/view/jmkang/about-our-lab). The project leverages a BLIP-2 like architecture with GPT-2 model as a language model.

## Model Architecture
We adapted a framework similar to the BLIP-2 model with the following components:
- **Transformer Decoder**: 4 layers, D = 512, 8 trainable query tokens.
- **Visual Features**: Pre-extracted by Baidu team, window size T = 30.
- **Language model**: GPT2-base and GPT-medium models


## Key Contributions of our solution
- End-to-end training with LLM and vision encoder instead of freezing the LLM weights.
- Changed the spotting action confidence task from binary to multi-class softmax classification.
- Discarded spotted actions with low confidences in localization.
- Combined generated captions from different models and removed overlapping actions.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/gladuz/soccernet-caption-delta.git
    cd soccernet-caption-delta
    ```
2. Create environment similar to [sn-caption baseline](https://github.com/SoccerNet/sn-caption/tree/main/Benchmarks/TemporallyAwarePooling):
    ```sh
    conda create -y -n soccernet-DVC python=3.8
    conda activate soccernet-DVC
    conda install -y pytorch torchvision torchtext pytorch-cuda -c pytorch -c nvidia
    pip install SoccerNet matplotlib scikit-learn spacy wandb
    pip install git+https://github.com/Maluuba/nlg-eval.git@master
    python -m spacy download en_core_web_sm
    pip install torchtext

    # only additional package is tiktoken and transformers used for GPT-2
    pip install tiktoken transformers
    ```

## Downloading dataset - refer to the [Soccernet's main captioning repository](https://github.com/SoccerNet/sn-caption)
First install `Soccernet` package using pip.
```bash
pip install SoccerNet
```
Then download the pre-extracted features from Baidu team. Note: the features seems to be at 1fps. We only use the extracted features in our solution.
```python
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
mySNdl = SNdl(LocalDirectory="path/to/SoccerNet")
mySNdl.downloadDataTask(task="caption-2024", split=["train","valid", "test","challenge"]) # SN challenge 2024
```


## Train the model
```bash
python main.py --SoccerNet_path=path/to/SoccerNet/  \
--model_name gpt2-train \
--GPU 0 \
--pool QFormer \
--gpt_type gpt2 \
--NMS_threshold 0.7
```
Replace `path/to/SoccerNet` with the local path for Soccernet dataset's `caption-2024` folder. Other options are listed in `main.py`. The results should be around 26 in METEOR.

## Main parameters
- `--gpt_type`: language model. `gpt2` or `gpt2-medium`. It will load the weights from huggingface by default.
- `--NMS_threshold`: discarding threshold for confidence levels. 
- `--NMS_window`: window of maximum supression. Only the highest confidence actions will be selected within that timeframe.

## Acknowledgements
Codebase is adopted from Soccernet team's baseline https://github.com/SoccerNet/sn-caption

GPT2 model code is from Andrej Karpathy's awesome [nanoGPT repo](https://github.com/karpathy/nanoGPT).

We would like to thank the SoccerNet team for organizing the challenge and providing the datasets. Special thanks to the Baidu team for the pre-extracted visual features.
