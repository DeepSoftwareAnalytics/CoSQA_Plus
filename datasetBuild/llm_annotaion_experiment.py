def merge_label():
    # models = ['gpt4o', 'gpt35', 'gpt351106']
    # models = ['llama370binstruct','claude3sonnet']
    # models = ['claude3opus']
    models = ["claude3sonnet1", "claude3sonnet2", "claude3sonnet3"]

    # 加载原始的query_code_pairs
    with open(
        "CoSQA-plus/dataset/human-label/human_query_code_pairs_1000.json", "r"
    ) as f:
        query_code_pairs = json.load(f)

    # 遍历每个模型
    for model in models:
        judgement_extraction(
            f"CoSQA-plus/dataset/human-label/human_query_code_pairs_1000_{model}.csv",
            f"CoSQA-plus/dataset/human-label/human_query_code_pairs_1000_{model}_processed.csv",
        )

        print(f"model: {model}")

        # 加载模型的标签数据
        df = pd.read_csv(
            f"CoSQA-plus/dataset/human-label/human_query_code_pairs_1000_{model}_processed.csv"
        )

        # 遍历query_code_pairs列表，并修改每个字典
        for index, pairs in enumerate(query_code_pairs):
            # 在df中查找与当前pairs['pair-index']匹配的行
            matching_rows = df[df["pair-index"] == pairs["pair-idx"]]

            # 如果找到了匹配的行，则添加新的键值对到pairs字典中
            if not matching_rows.empty:
                judgement = matching_rows["judgement"].values[0]
                query_code_pairs[index][model] = 1 if judgement == "yes" else 0
    print("start to save...")
    with open(
        "CoSQA-plus/dataset/human-label/human_query_code_pairs_1000_merge5.json", "w"
    ) as f:
        json.dump(query_code_pairs, f)
    json_to_csv(
        "CoSQA-plus/dataset/human-label/human_query_code_pairs_1000_merge5.json",
        "CoSQA-plus/dataset/human-label/human_query_code_pairs_1000_merge5.csv",
    )
