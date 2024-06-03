import krippendorff
import numpy as np
import pandas as pd
import math

def calculate_krippendorff_alpha(df,columns_list,model_name):
        # 挑选df指定列
    df_selected = df[columns_list]
    # 计算Krippendorff's alpha
    data = df_selected.T.values.tolist()
    data_tuple = tuple(' '.join(map(str, row)) for row in data)
    newlistconvert =[[np.nan if (v == "*" or v=="N/A" or math.isnan(float(v))) else int(float(v)) for v in coder.split()] for coder in data_tuple]
    alpha = krippendorff.alpha(reliability_data=newlistconvert, level_of_measurement='nominal')
    print(f"nominal krippendorff alpha 3 huamn with model {model_name} : {alpha}")
    # alpha = krippendorff.alpha(reliability_data=newlistconvert, level_of_measurement='interval')
    # print(f"interval krippendorff alpha 3 huamn with model {model_name} : {alpha}")

def main():
    df = pd.read_excel("CoSQA-plus/datasetBuild/human_query_code_pairs_1000_merge_all5.xlsx")
    # columns = "gpt4-turbo gpt4o gpt35 gpt351106 llama370binstruct claude3sonnet deepseek claude3haiku claude3opus Miking-exact Miking-50% Cookie-label-exact Cookie-label-50% thinkerhui-exact thinkerhui-50%"
    # columns_list = columns.split()
    columns_list = ["claude3sonnet","claude3sonnet1","claude3sonnet2","claude3sonnet3"]
    columns_list1 = ["Miking-exact","Cookie-label-exact","thinkerhui-exact"]
    # columns_list1 = ["claude3sonnet","claude3haiku"]
    columns_list2 = ["Miking-50%","Cookie-label-50%","thinkerhui-50%"]
    print("nominal krippendorff alpha 3 huamn(exact match)")
    for i in range(4):
        columns_list_temp = columns_list1.copy()
        columns_list_temp.append(columns_list[i])
        calculate_krippendorff_alpha(df,columns_list_temp,columns_list[i])
    print("nominal krippendorff alpha 3 huamn(50% match),notice:model still exact match")
    for i in range(4):
        columns_list_temp = columns_list2.copy()
        columns_list_temp.append(columns_list[i])
        calculate_krippendorff_alpha(df,columns_list_temp,columns_list[i])
    print("exact human just")
    calculate_krippendorff_alpha(df,columns_list1,"human only")
    print("50% human just")
    calculate_krippendorff_alpha(df,columns_list2,"human only")
    print("claude3 sonnet only")
    calculate_krippendorff_alpha(df,columns_list,"claude3 sonnet only") 
if __name__ == "__main__":
    main()