# Evaluation

## MCQ Benchmarks

### Introduction
The benchmark we developed **Fundus-MMBench**: [MeteorElf/Fundus-MMBench](https://huggingface.co/datasets/MeteorElf/Fundus-MMBench) is hosted on HuggingFace. This benchmark is **for academic research only**. By applying for access, you agree to these terms.

You can run the evaluation on Fundus-MMBench using [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit). Note that Fundus-MMBench(tsv version) is not officially supported, but can be regarded as a Custom MCQ dataset.

**GMAI-MMBench(fundus image subset)**: [uni-medical/GMAI-MMBench](https://github.com/uni-medical/GMAI-MMBench)

GMAI-MMBench can be evaluated directly using VLMEvalKit, but manual screening of the fundus image subset may be required.

### API Model
For the evaluation of models that require calling APIs, you can use the API method in [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) or use `run.sh` in `src/eval/eval_api`.

## Open-domain Tasks

###  Localization Ability


###  Clinical Consistency

Existing likelihood-based benchmarks for medical text generation, such as BLEU and ROUGE, inadequately assess semantic plausibility. To overcome this, we introduce a multi-granularity semantic matching framework that evaluates the accuracy of generated medical reports. This framework leverages a VLM(GPT-4o), to perform a structured evaluation of clinical logical consistency. 

After the model generates a medical report, `src/eval/eval_open/generation/prompt_ref.txt` is used to let VLM evaluate and score the generated content with the benchmark labels.
