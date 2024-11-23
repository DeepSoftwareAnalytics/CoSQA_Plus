import json
import random
import pandas as pd
from collections import Counter

from matplotlib import pyplot as plt
import pickle
import re
from tqdm import tqdm

DATASET_PATH = "CoSQA_Plus/dataset/"
CSN_ORIGIN_JSON_FILE = DATASET_PATH+"python_dedupe_definitions_v2.json"
CSN_CODE_FILE = DATASET_PATH+"CSN-code.json"
STAQC_CODE_FILE = DATASET_PATH+"StaQC-code.json"
CSN_STAQC_CODE_FILE = DATASET_PATH+"CSN-StaQC-code.json"
QUERY_FILE = DATASET_PATH+"query.json"
COSQA_PAIRS_FILE = DATASET_PATH+"cosqa-all.json"


def build_augment_dataset():
    """
    merge origin_pairs and augment_pairs,
    finally, get augment dataset
    """
    ORIGIN_PAIRS_JSON_FILE = (
        DATASET_PATH+"stage_2_final/query_code_pairs_annotated.json"
    )
    ORIGIN_CODE_JSON_FILE = DATASET_PATH+"stage_2_final/codebase.json"
    AUGMENT_CODEBASE_JSON_FILE = (
        DATASET_PATH+"stage_3_augment/final_augment_codebase.json"
    )
    AUGMENT_PAIRS_JSON_FILE = (
        DATASET_PATH+"stage_3_augment/final_augment_query_code_pairs.json"
    )

    with open(ORIGIN_PAIRS_JSON_FILE, "r") as f:
        origin_pairs_json = json.load(f)
    with open(AUGMENT_PAIRS_JSON_FILE, "r") as f:
        augment_pairs_json = json.load(f)
    with open(ORIGIN_CODE_JSON_FILE, "r") as f:
        origin_codebase_json = json.load(f)
    # build augment codebase
    print(f"build augment codebase...")
    augment_codebase_json = origin_codebase_json
    augment_code_dict = dict()
    code_idx = len(origin_codebase_json)
    print(f"build origin codebase done...len:{len(augment_codebase_json)}")
    for item in augment_pairs_json:
        augment_codebase_json.append({"code-idx": code_idx, "code": item["code"]})
        augment_code_dict[item["code"]] = code_idx
        code_idx += 1
    print(f"build augment codebase done...len:{len(augment_codebase_json)}")
    with open(AUGMENT_CODEBASE_JSON_FILE, "w") as f:
        json.dump(augment_codebase_json, f)
    # build augment pairs
    print(f"build augment pairs...")
    augment_pairs_json = []
    for item in origin_pairs_json:
        augment_pairs_json.append(
            {
                "pair-idx": item["pair-idx"],
                "query-idx": item["query-idx"],
                "query": item["query"],
                "code-idx": item["code-idx"],
                "code": item["code"],
                "label": item["label"],
            }
        )
    pair_idx = len(origin_pairs_json)
    for item in augment_pairs_json:
        augment_pairs_json.append(
            {
                "pair-idx": pair_idx,
                "query-idx": item["query-index"],
                "query": item["query"],
                "code-idx": augment_code_dict[item["code"]],
                "code": item["code"],
                "label": 1,
            }
        )
        pair_idx += 1
    print(f"build augment pairs done...len:{len(augment_pairs_json)}")
    with open(AUGMENT_PAIRS_JSON_FILE, "w") as f:
        json.dump(augment_pairs_json, f)


def from_top5_to_individual():
    """
    process the top5 selected code data to create individual code-query pairs
    """
    SELECTED_CODE_JSON_FILE = DATASET_PATH+"selected_code.json"
    CANDIDATE_PAIRS_JSON_FILE = DATASET_PATH+"query_code_pair.json"
    with open(SELECTED_CODE_JSON_FILE, "r") as f:
        top5_code_data = json.load(f)
    with open(CSN_STAQC_CODE_FILE, "r") as f:
        all_code_data = json.load(f)
    code_data = []
    pair_index = 0
    for idx, item in enumerate(top5_code_data):
        code_index_list = item["top_code_index"][1:-1].strip().split(",")
        for code_index in code_index_list:
            code_index = int(code_index)
            code_data.append(
                {
                    "pair-index": pair_index,
                    "query-index": idx,
                    "query": item["query"],
                    "code-index": code_index,
                    "code": all_code_data[code_index]["code"],
                }
            )
            pair_index += 1
    print(len(code_data))
    with open(CANDIDATE_PAIRS_JSON_FILE, "w") as f:
        json.dump(code_data, f)


def judgement_extraction(input_file, processed_file):
    df = pd.read_csv(input_file, index_col=0)
    # 从answer列中提取origin_answer
    # 使用正则表达式匹配 "judgement": "([^"]+)"
    count_yes = 0
    count_no = 0
    for index, row in df.iterrows():
        data_str = row["origin_answer"]
        match = re.search(r'("judgement"|"judgment"): "([^"]+)"', data_str)
        if match:
            judgement_value = match.group(2)
            df.at[index, "judgement"] = judgement_value.lower()
            if judgement_value == "yes":
                count_yes += 1
            elif judgement_value == "no":
                count_no += 1
            else:
                print(f"No match found for pair-index {index}")
            # print(judgement_value)
        else:
            print(f"No match found for pair-index {index}")
            print(data_str)
    print(f"num of yes: {count_yes}")
    print(f"num of no: {count_no}")
    df.to_csv(processed_file)


def judgement_extraction1(input_file1, processed_file1):
    df = pd.read_csv(input_file1, index_col=0)
    # 从answer列中提取origin_answer
    # 使用正则表达式匹配 "judgement": "([^"]+)"
    for index, row in df.iterrows():
        data_str = str(row["judgement"])
        if "yes" in data_str and "no" in data_str:
            df.at[index, "final_judgement"] = ""
        elif "yes" in data_str:
            df.at[index, "final_judgement"] = "yes"
        elif "no" in data_str:
            df.at[index, "final_judgement"] = "no"
        else:
            df.at[index, "final_judgement"] = ""
            print(f"No match found for pair-index {index}")
    df.to_csv(processed_file1)


def build_dataset_from_pairs():
    '''
    build dataset(codebase + pairs based on the codebase)
    from query-code pairs 
    
    '''
    with open(
        DATASET_PATH+"query_code_pair_label.json", "r"
    ) as f:
        code_pair_label = json.load(f)
    # label_data_df = pd.read_csv(DATASET_PATH+"dataset_annotation_GPT4_processed.csv")
    code_counter = Counter()
    # combine same code
    for item in code_pair_label:
        code_counter.update([item["code"].strip()])
    # traverse code_counter, build codebase
    print("Codebase building...")
    code_data = dict()
    code_base = []
    for idx, code in enumerate(code_counter.keys()):
        code_data[code] = {
            "code": code,
            "code-idx": idx,
        }
        code_base.append(
            {
                "code-idx": idx,
                "code": code,
            }
        )
    print("Codebase building done.")
    # build final_query-code-pairs
    print("Final query-code-pairs building...")
    final_query_code_pair_label = []
    for item in code_pair_label:
        final_query_code_pair_label.append(
            {
                "pair-idx": item["pair-index"],
                "query-idx": item["query-index"],
                "query": item["query"],
                "code-idx": code_data[item["code"].strip()]["code-idx"],
                "code": item["code"],
                # 'label':label_data_df[label_data_df['pair-index']==item['pair-index']]['judgement'].values[0],
            }
        )
    print("Final query-code-pairs building done.")
    print(f"codebase:{len(code_data)}")
    print(f"query-code-pairs:{len(final_query_code_pair_label)}")
    # save to json file
    with open(
        DATASET_PATH+"final_query_code_pairs.json", "w"
    ) as f:
        json.dump(final_query_code_pair_label, f)
        print(f"final_query_code_pairs.json saved: {len(final_query_code_pair_label)}")
    with open(DATASET_PATH+"codebase.json", "w") as f:
        json.dump(code_base, f)
        print(f"codebase.json saved: {len(code_base)}")


def divide_dataset():
    with open(
        DATASET_PATH+"stage_3_augment/final_augment_query_code_pairs.json", "r"
    ) as f:
        augment_query_code_pairs = json.load(f)
    augment_query_code_pairs_for_search = []
    for item in augment_query_code_pairs:
        if item["label"] == 1:
            augment_query_code_pairs_for_search.append(item)
    with open(
        DATASET_PATH+"stage_3_augment/final_augment_query_code_pairs_for_search.json",
        "w",
    ) as f:
        json.dump(augment_query_code_pairs_for_search, f)


def pairs_transform():
    print("start to load...")
    df = pd.read_csv(
        DATASET_PATH+"stage_2_final/dataset_annotation_claude3sonnet_processed.csv"
    )
    codebase = pd.read_json(DATASET_PATH+"stage_2_final/codebase.json")
    query = pd.read_json(DATASET_PATH+"query.json")
    query_code_pairs = []
    for idx, item in tqdm(df.iterrows(), total=len(df)):
        judgement = item["judgement"]
        # Convert the corresponding lines of query and codebase into dictionaries.
        query_row = query[query["query-idx"] == item["query-index"]].to_dict(
            orient="records"
        )
        code_row = codebase[codebase["code-idx"] == item["code-index"]].to_dict(
            orient="records"
        )

        # ensure that each query and code row has exactly one record
        if len(query_row) == 1 and len(code_row) == 1:
            query_code_pairs.append(
                {
                    "pair-idx": item["pair-index"],
                    "query-idx": item["query-index"],
                    "query": query_row[0]["query"],
                    "code-idx": item["code-index"],
                    "code": code_row[0]["code"],
                    "label": 1 if judgement == "yes" else 0,
                }
            )
        else:
            print(
                f"Warning: Incorrect number of records found for query_idx {item['query-index']} or code_idx {item['code-index']}."
            )
    with open(
        DATASET_PATH+"stage_2_final/query_code_pairs_annotated.json", "w"
    ) as f:
        json.dump(query_code_pairs, f)


def divide_query_by_label(pairs_csv_file):
    pairs_data = pd.read_csv(pairs_csv_file)
    yes_label_query_data = pairs_data[pairs_data["judgement"] == "yes"]
    yes_label_query = []
    for idx, row in yes_label_query_data.iterrows():
        yes_label_query.append(row["query-index"])
    yes_label_query = set(yes_label_query)
    no_label_query = []
    for idx, row in pairs_data.iterrows():
        if row["query-index"] not in yes_label_query:
            no_label_query.append(row["query-index"])
    no_label_query = set(no_label_query)
    return yes_label_query, no_label_query