#!/bin/bash

################# Execute them one by one in sequence. #################

## Calling the API to generate the answer
## The image path in the benchmark is a local path, not base64 encoded.

# python call_api.py \
#   --benchmark_tsv /path/to/xxx_local.tsv \
#   --model_name gpt-4o \
#   --base_url http://xxxx \
#   --api_key sk-xxxxxxxx


# ## Re-call the API if the model fails to respond.
# python recall_api.py \
#   --input_csv /result/of/call_api \
#   --benchmark_tsv /path/to/xxx_local.tsv \
#   --output_csv /save/result/of/re-calling/API \
#   --model_name gpt-4o \
#   --base_url http://xxxx \
#   --api_key sk-xxxxxxxx


## Computes hit rate and accuracy, and reports the accuracy for each category in the order specified.
python get_acc.py \
  --gpt_response_csv /output_csv/of/recall_api \
  --benchmark_tsv /path/to/xxx_local.tsv \
  --output_csv /save/result/of/acc

