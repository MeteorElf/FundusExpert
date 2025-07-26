# <img src="asset/logo.png" alt="FundusExpert Logo" width="70" /> FundusExpert
This repository is the official implementation of the paper **Constructing Ophthalmic MLLM for Positioning-diagnosis Collaboration Through Clinical Cognitive Chain Reasoning**.

[🤗 Model](https://huggingface.co/MeteorElf/FundusExpert) | [🤗 Dataset](https://huggingface.co/datasets/MeteorElf/Fundus-MMBench) | [📝 Paper](https://huggingface.co/papers/2507.17539) | [📖 arXiv](https://arxiv.org/abs/2507.17539)

## Introduction

This paper introduces FundusExpert, an ophthalmology-specific MLLM with integrated positioning-diagnosis reasoning capabilities, along with FundusGen, a dataset constructed through the intelligent Fundus-Engine system.

## Data Processing

## Setup

Clone this repository and install the dependencies.

Please refer to [InternVL Installation](https://internvl.readthedocs.io/en/latest/get_started/installation.html) or use the `src/internvl25_requirements.txt` to build the environment.

## Model

Our model weights are hosted on HuggingFace and require application to access. This model weights are for academic research only. By applying for access, you agree to these terms.

**Model**: [MeteorElf/FundusExpert](https://huggingface.co/MeteorElf/FundusExpert)

### Quick Start

Inference with single GPU:

`python src/quick_start.py`

## Evaluation

Our benchmark is hosted on HuggingFace and require application to access. This benchmark is for academic research only. By applying for access, you agree to these terms.

**Fundus-MMBench**: [MeteorElf/Fundus-MMBench](https://huggingface.co/datasets/MeteorElf/Fundus-MMBench)

You can run the evaluation on Fundus-MMBench using [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit). Note that Fundus-MMBench(tsv version) is not officially supported, but can be regarded as a Custom MCQ dataset.

For the evaluation of models that require calling APIs, you can use the API method in [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) or use the `src/eval/eval_api` code.

**GMAI-MMBench(fundus image subset)**: [uni-medical/GMAI-MMBench](https://github.com/uni-medical/GMAI-MMBench)


## Contact
Xinyao Liu: liuxinyao@mail.ustc.edu.cn

Diping Song: songdiping@pjlab.org.cn

## Acknowledgements

Our model is based on [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL). Our evaluation code is based on [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit). We would like to thank their excellent work and open source contributions.

