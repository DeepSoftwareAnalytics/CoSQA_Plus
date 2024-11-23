import json
def from_top5_to_individual_multimodel(model_name=""):
    with open(
        f"CoSQA-plus/dataset/multi-model/{model_name}_selected_code.json", "r"
    ) as f:
        top5_code_data = json.load(f)
    with open("CoSQA-plus/dataset/stage_1_origin/CSN-StaQC-code.json", "r") as f:
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
    with open(
        f"CoSQA-plus/dataset/multi-model/{model_name}_query_code_pair_label.json", "w"
    ) as f:
        json.dump(code_data, f)

def merge_query_code_pair():
    """
    将query和code的pair合并
    query-idx和code-idx相同则合并
    生成这个是为了打标签
    """
    models = ["unixcoder-base", "codebert-base", "codet5p-110m-embedding"]
    all_pairs_json = []
    # 汇集合并每个模型的query-code pairs
    for model in models:
        with open(
            f"CoSQA-plus/dataset/{model}_query_code_pair_label_1000.json", "r"
        ) as f:
            pairs_json = json.load(f)
        for item in pairs_json:
            new_item = {
                "query-idx": item["query-index"],
                "query": item["query"].strip(),
                "code": item["code"].strip(),
                "code-idx": item["code-index"],
            }
            if new_item not in all_pairs_json:
                all_pairs_json.append(new_item)
    print(f"merge pairs num before remove: {len(all_pairs_json)}")
    # 在origin_pairs中存在的pairs也排除掉
    with open("CoSQA-plus/dataset/query_code_pair_label.json", "r") as f:
        origin_pairs_json = json.load(f)
    for item in origin_pairs_json:
        new_item = {
            "query-idx": item["query-index"],
            "query": item["query"].strip(),
            "code": item["code"].strip(),
            "code-idx": item["code-index"],
        }
        if new_item in all_pairs_json:
            print(f"remove {new_item}")
            all_pairs_json.remove(new_item)
    print(f"merge pairs num after remove: {len(all_pairs_json)}")
    all_pairs_json_with_idx = []
    for idx, item in enumerate(all_pairs_json):
        all_pairs_json_with_idx.append(
            {
                "pair-index": idx,
                "query-index": item["query-idx"],
                "query": item["query"],
                "code": item["code"],
                "code-index": item["code-idx"],
            }
        )

    with open("CoSQA-plus/dataset/1000_query_code_pair_merge.json", "w") as f:
        json.dump(all_pairs_json_with_idx, f)

def build_codebase_multimodel(model_name=""):
    with open(
        f"CoSQA-plus/dataset/multi-model/{model_name}_query_code_pair_label.json", "r"
    ) as f:
        code_pair_label = json.load(f)
    # label_data_df = pd.read_csv("CoSQA-plus/dataset/dataset_annotation_GPT4_processed.csv")
    code_counter = Counter()
    # 这个合并重复的code
    for item in code_pair_label:
        code_counter.update([item["code"].strip()])
    # 遍历code_counter,构建codebase
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
    # 构建最终的query-code-pairs
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
    # 保存为json
    with open(
        f"CoSQA-plus/dataset/multi-model/{model_name}_final_query_code_pairs.json", "w"
    ) as f:
        json.dump(final_query_code_pair_label, f)
        print(f"final_query_code_pairs.json saved: {len(final_query_code_pair_label)}")
    with open(f"CoSQA-plus/dataset/multi-model/{model_name}_code_data.json", "w") as f:
        json.dump(code_base, f)
        print(f"codebase.json saved: {len(code_base)}")

def multi_model_process():
    models = ["unixcoder-base", "codebert-base", "codet5p-110m-embedding"]
    for model_name in models:
        print(f"Processing {model_name}...")
        from_top5_to_individual_multimodel(model_name)
        build_codebase_multimodel(model_name)
        print(f"Processing {model_name} done.")
def select_pairs_1000_query():
    """
    随机筛选出1000条query并调用from_top5_to_individual
    """
    models = ["unixcoder-base", "codebert-base", "codet5p-110m-embedding"]
    with open("CoSQA-plus/dataset/query_1000.json", "r") as f:
        query_json = json.load(f)
    for model in models:
        with open(f"{model}_selected_code4.json", "r") as f:
            pairs_json = json.load(f)
        # 筛选出1000条query对应的pairs
        pairs_json = [
            item
            for item in pairs_json
            if item["query"] in [item["query"] for item in query_json]
        ]
        with open(f"CoSQA-plus/dataset/{model}_selected_code_1000.json", "w") as f:
            json.dump(pairs_json, f)
        print(f"{model} selected query num: {len(pairs_json)}")
        # 转变形式
        from_top5_to_individual_multimodel(model)
