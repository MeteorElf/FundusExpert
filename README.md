# FundusExpert
This repository is the official implementation of the paper **Constructing Ophthalmic MLLM for Positioning-diagnosis Collaboration Through Clinical Cognitive Chain Reasoning**.

[🤗 Model](https://huggingface.co/MeteorElf/FundusExpert) | [🤗 Dataset](https://huggingface.co/datasets/MeteorElf/Fundus-MMBench) | [📝 Paper](https://huggingface.co/papers/2507.17539) | [📖 arXiv](https://arxiv.org/abs/2507.17539)

## Introduction

<img src="asset/demo.png" alt="FundusExpert demo" width="700" />

This paper introduces FundusExpert, an ophthalmology-specific MLLM with integrated positioning-diagnosis reasoning capabilities, along with FundusGen, a dataset constructed through the intelligent Fundus-Engine system.

## Data Processing

[📌 Data Processing](./data_processing/data_processing.md)

## Setup

Clone this repository and install the dependencies.

Please refer to [InternVL Installation](https://internvl.readthedocs.io/en/latest/get_started/installation.html) or use the `src/internvl25_requirements.txt` to build the environment.

### Quick Start

Inference with single GPU:

`python src/quick_start.py`

## Evaluation

[📌 Evaluation](./src/eval/eval.md)

## Contact
Xinyao Liu: liuxinyao@mail.ustc.edu.cn

Diping Song: songdiping@pjlab.org.cn

## Acknowledgements

Our model is based on [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL). Our evaluation code is based on [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit). We would like to thank their excellent work and open source contributions.

