import json
import pandas as pd
from process_data import QUERY_FILE
import tqdm
def relabel_cosqa():
    with open("CoSQA-plus/dataset/CoSQA/cosqa-retrieval-train-19604.json", "r") as f:
        cosqa = json.load(f)
    new_label = pd.read_excel(
        "CoSQA-plus/dataset/CoSQA/cosqa_all_gpt4_match_zero_shot_cot_processed1.xlsx"
    )

    for index, item in enumerate(cosqa):
        new_judgement = new_label.loc[
            new_label["idx"] == item["idx"], "judgement"
        ].values[0]
        if new_judgement == "yes":
            cosqa[index]["label"] = 1
        elif new_judgement == "no":
            cosqa[index]["label"] = 0
    with open(
        "CoSQA-plus/dataset/CoSQA/cosqa-retrieval-train-19604_new.json", "w"
    ) as f:
        json.dump(cosqa, f)


def get_claude_code(no_label_query):
    """
    read pairs in which the code generated by Claude3,
    and filter out pairs in which the query does not have matched code.

    :param no_label_query: a list of indices corresponding to queries that do not have matched code.
    These indices are used to filter out query-code pairs
    """
    CLAUDE3_CODE_FILE = "CoSQA-plus/dataset/stage_3_augment/claude3_code.xlsx"
    CLAUDE3_AUGMENT_PAIRS_FILE = "CoSQA-plus/dataset/Claude3_query_code_pairs.json"

    df = pd.read_excel(CLAUDE3_CODE_FILE)
    with open(QUERY_FILE) as f:
        query_json = json.load(f)
    # 筛选出没有label为yes的query对应的code,组成query-code pairs
    pairs_data = []
    for query_index in no_label_query:
        query = query_json[query_index]["query"]
        pairs_data.append(
            {
                "pair-index": query_index,
                "query-index": query_index,
                "query": query,
                "code": df[df["query"] == query]["code"].values[0],
            }
        )
    print(f"claude3 query-code pairs num:{len(pairs_data)}")
    with open(CLAUDE3_AUGMENT_PAIRS_FILE, "w") as f:
        json.dump(pairs_data, f)
        
        
def filter_codebase():
    
    with open(
        "CoSQA-plus/dataset/stage_3_augment/final_augment_codebase.json", "r"
    ) as f:
        codebase = json.load(f)

    with open(
        "CoSQA-plus/dataset/stage_3_augment/final_augment_query_code_pairs.json", "r"
    ) as f:
        query_code_pairs = json.load(f)
    new_codebase = []
    # 只保留在query_code_pairs中出现的code
    for item in tqdm(codebase):
        for pair in query_code_pairs:
            if pair["label"] == 1 and item["code-idx"] == pair["code-idx"]:
                new_codebase.append(item)
                break
    print(f"new codebase len:{len(new_codebase)}")
    with open(
        "CoSQA-plus/dataset/stage_3_augment/final_augment_codebase_label1.json", "w"
    ) as f:
        json.dump(new_codebase, f)