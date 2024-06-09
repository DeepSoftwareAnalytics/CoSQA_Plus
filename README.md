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

format example:

```json
{
    "code-idx": 0,
    "code": "def clean_empty(d):\n    if not isinstance(d, (dict, list)):\n        return d\n    if isinstance(d, list):\n        return [v for v in (clean_empty(v) for v in d) if v]\n    return {k: v for k, v in ((k, clean_empty(v)) for k, v in d.items()) if v}"
}
```

CoSQA+/CoSQA query base: `query.json`

format example:

```json
{	
	"query-idx": 0, 
	"query": "python remove all empty items in list"
}
```

CoSQA+ query-code pairs: `final_augment_query_code_pairs.json`

format example:

```json
{
    "pair-idx": 0,
    "query-idx": 0,
    "query": "python remove all empty items in list",
    "code-idx": 0,
    "code": "def clean_empty(d):\n    if not isinstance(d, (dict, list)):\n        return d\n    if isinstance(d, list):\n        return [v for v in (clean_empty(v) for v in d) if v]\n    return {k: v for k, v in ((k, clean_empty(v)) for k, v in d.items()) if v}",
    "label": 1
}
```

CoSQA+ query-code pairs (filtered for label 1): `final_augment_query_code_pairs_for_search.json`

### Code

The `datasetBuild` folder stores the code for building the CoSQA+ dataset and testing large model question answering.

Code for tool functions to build the CoSQA+ dataset: `process_data.py`

Program for matching queries to code: `select_code.py`

Program for calculating Krippendorff’s alpha: `krippendorff_calculate.py`


The `evaluateMMRR` folder stores the code for testing code search models and methods on the CoSQA+ benchmark.

Test program for pylucene: `run_MMRR_pylucene.py`

Test program for bow: `bow_MMRR.py`

Test program for transformer large models (CodeBERT, UniXcoder, CodeT5+ etc.): `run_MMRR_transformer.py`


The `prompt` folder stores prompt texts.

## CoSQA+ Construction

Note that fully reproducing this part requires significant resources, especially since annotating with large models incurs high costs. Therefore, we mainly provide guidelines for reproducing query-to-code matching using a multi-model approach.

### Download Code Datasets

**The quickest way** is to download our curated datasets (Google Drive):

Filtered and merged StaQC Python code `StaQC-code.json`:

https://drive.google.com/file/d/1rgA4ptcUBioHbK9T49T2GYQ8A5Jde2nb/view?usp=sharing

Filtered StaQC Python code and CodeSearchNet Python code merged dataset `CSN-StaQC-code.json`:

https://drive.google.com/file/d/15n8H3WzfjC0MejXvwU2o7seRI0tjQ1wk/view?usp=drive_link

If you want to reconstruct the codebase, you can follow the instructions below:

We need `python_dedupe_definitions_v2.pkl` from CodeSearchNet's python.zip and two pkl files from StaQC: `python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle` and `python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_code.pickle`.

After downloading, please place them in the `dataset` folder of the project.

The code datasets from [CodeSearchNet](https://github.com/github/CodeSearchNet/tree/master?tab=readme-ov-file) and [StaQC](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset) can be downloaded according to their official instructions.

Alternatively, you can download them via Hugging Face:

[code-search-net/code_search_net at main (huggingface.co)](https://huggingface.co/datasets/code-search-net/code_search_net/tree/main/data)

[koutch/staqc · Datasets at Hugging Face](https://huggingface.co/datasets/koutch/staqc)

For StaQC, preliminary filtering is needed:

```python
python datasetBuild/StaQC_data_to_json.py 
```

Note that the parameter for the function `check_pickle_file("python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle")` should be changed to the directory where you stored the StaQC Python code pkl files. Then, merge the two JSON files.

Finally, merge the CSN Python code with the StaQC Python code. This can be done by calling `process_CSN()` and `merge_CSN_and_StaQC()` in `datasetBuild/process_data.py`.

### Download Query Dataset

[CoCLR/data/qa/cosqa-all.json at main · Jun-jie-Huang/CoCLR (github.com)](https://github.com/Jun-jie-Huang/CoCLR/blob/main/data/qa/cosqa-all.json)

Place the downloaded `cosqa-all.json` in the `dataset` folder.

Call `process_query()` in `datasetBuild/process_data.py` to complete the processing and obtain `query.json`.

### Match Code for Queries

The main task of matching code to queries is done using `datasetBuild/select_code.py`.
