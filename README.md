# <img src="asset/logo.png" alt="FundusExpert Logo" width="70" /> FundusExpert
Constructing Ophthalmic MLLM for Positioning-diagnosis Collaboration Through Clinical Cognitive Chain Reasoning.

## Introduction

This paper introduces FundusExpert, an ophthalmology-specific MLLM with integrated positioning-diagnosis reasoning capabilities, along with FundusGen, a dataset constructed through the intelligent Fundus-Engine system.

## Setup

Clone this repository and install the dependencies.

Please refer to [InternVL Installation](https://internvl.readthedocs.io/en/latest/get_started/installation.html) or use the `src/internvl25_requirements.txt` to build the environment.

## Data and Models

Our model weights and benchmarks are hosted on HuggingFace and require application to access. This model weights and benchmarks are for academic research only. By applying for access, you agree to these terms.

**Model**: [MeteorElf/FundusExpert](https://huggingface.co/MeteorElf/FundusExpert)

**Please send an email to `liuxinyao@mail.ustc.edu.cn` and `songdiping@pjlab.org.cn`**. Please include your HuggingFace username and a brief self-introduction in the email. We will authorize you as soon as possible.

### Quick Start

Inference with single GPU:

`python src/quick_start.py`

## Evaluation


## Acknowledgements

Our model is based on the [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL). We would like to thank its authors for their excellent work and open source contributions.
