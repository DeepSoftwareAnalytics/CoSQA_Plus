import krippendorff
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def calculate_krippendorff_alpha(df, columns_list, model_name, num_pairs):
    df_selected = df[columns_list].iloc[:num_pairs]
    data = df_selected.T.values.tolist()
    data_tuple = tuple(' '.join(map(str, row)) for row in data)
    newlistconvert = [
        [np.nan if (v == "*" or v == "N/A" or math.isnan(float(v))) else int(float(v))
         for v in coder.split()] for coder in data_tuple
    ]
    alpha = krippendorff.alpha(reliability_data=newlistconvert, level_of_measurement='nominal')
    return round(alpha, 3)

def main():
    # 读取文件
    df = pd.read_excel("CoSQA-plus/dataset/human-label/all_merge/human_query_code_pairs_1000_merge_all6.xlsx")
    columns = "gpt4-turbo gpt4o gpt35 gpt351106 llama370binstruct claude3sonnet deepseek claude3haiku claude3opus Miking-exact Miking-50% Cookie-label-exact Cookie-label-50% thinker-exact thinker-50%"
    columns_list = columns.split()
    columns_list1 = ["Miking-exact", "Cookie-label-exact", "thinker-exact"]
    columns_list2 = ["Miking-50%", "Cookie-label-50%", "thinker-50%"]

    alphas_exact = []
    alphas_50 = []

    for i in range(2, 101):  # 0-indexed, so we start at 1 to get 100 pairs
        alphas_exact.append(calculate_krippendorff_alpha(df, columns_list1, "human only", i))
        alphas_50.append(calculate_krippendorff_alpha(df, columns_list2, "human only", i))

    plt.figure(figsize=(10, 5))
    plt.plot(range(2, 101), alphas_exact, label="Exact Match")
    plt.plot(range(2, 101), alphas_50, label="50% Match")
    plt.xlabel("Number of Pairs")
    plt.ylabel("Alpha Value")
    plt.title("Krippendorff's Alpha vs Number of Pairs")
    plt.legend()
    # plt.show()
    plt.savefig("CoSQA-plus/dataset/human-label/all_merge/krippendorff_alpha_human_only.png")

if __name__ == "__main__":
    main()