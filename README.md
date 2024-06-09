# CoSQA-plus

This repository contains code and datasets for the paper "CoSQA+:"

Our primary work can be divided into three parts: constructing CoSQA+, testing large models for question answering, and testing code search models and methods. This code repository will provide the corresponding code for these three sections as well as the data required to reproduce the results.

The construction of CoSQA+ can be broken down into three steps. The first step is data collection and processing. The second involves matching queries with code to form 100K pairs, and using Claude 3 Sonnet to judge whether the code matches the query. The third step is generating code for queries that were not successfully matched with code.

Testing code search models and methods requires downloading and configuring various code search models and methods to perform code search tests, calculating MRR and MMRR.

Testing large models for question answering primarily focuses on evaluating the performance of various large models in question answering, calculating their Krippendorff’s alpha compared to human annotators and their accuracy relative to standards.

Experimen Environment:

Hardware:

- Intel(R) Xeon(R) Platinum 8360H CPU @ 3.00GHz
- 4 * NVIDIA GeForce RTX 4090  
- 500GB RAM

Software:

- Ubuntu 22.04.1 LTS
- Python3.10.6

Requirements:

```
numpy==1.26.4
transformers==4.39.1
tqdm==4.66.2
scikit-learn==1.4.2
pandas==2.2.1
ollama==0.1.7
openai==1.30.5
matplotlib==3.8.4
joblib==1.4.0
krippendorff==0.6.1
```

## Dataset and Code Overview

### Datasets

The datasets are currently available for viewing at the following Google Drive link:

https://drive.google.com/drive/folders/1yoIoNfVI4vN5dk3VLuvGGorOE4fOleDT?usp=sharing

CoSQA+ codebase: `final_augment_codebase.json`

CoSQA+/CoSQA query base: `query.json`

CoSQA+ query-code pairs: `final_augment_query_code_pairs.json`

CoSQA+ query-code pairs (filtered for label 1): `final_augment_query_code_pairs_for_search.json`

### Code

The `datasetBuild` folder stores the code for building the CoSQA+ dataset and testing large model question answering.

Code for tool functions to build the CoSQA+ dataset: [process_data.py](https://github.com/thinkerhui/CoSQA-plus/blob/main/datasetBuild/process_data.py)

Program for matching queries to code: [select_code.py](https://github.com/thinkerhui/CoSQA-plus/blob/main/datasetBuild/select_code.py)

Program for calculating Krippendorff’s alpha: [krippendorff_calculate.py](https://github.com/thinkerhui/CoSQA-plus/blob/main/datasetBuild/krippendorff_calculate.py)


The `evaluateMMRR` folder stores the code for testing code search models and methods on the CoSQA+ benchmark.

Test program for pylucene: [run_MMRR_pylucene.py](https://github.com/thinkerhui/CoSQA-plus/blob/main/evaluateMMRR/run_MMRR_pylucene.py)

Test program for bow: [bow_MMRR.py](https://github.com/thinkerhui/CoSQA-plus/blob/main/evaluateMMRR/bow_MMRR.py)

Test program for transformer large models (CodeBERT, UniXcoder, CodeT5+ etc.): [run_MMRR_transformer.py](https://github.com/thinkerhui/CoSQA-plus/blob/main/evaluateMMRR/run_MMRR_transformer.py)


The `prompt` folder stores prompt texts.
