import krippendorff
import numpy as np
import pandas as pd
import math

def calculate_krippendorff_alpha(df, columns_list, model_name):
    # 挑选df指定列
    df_selected = df[columns_list]
    
    # 获取df的总行数
    total_rows = len(df_selected)
    
    # 存储所有计算结果的列表
    alpha_list = []
    
    # 每 50 行计算一次 Krippendorff's alpha
    for start_row in range(0, total_rows, 100):
        end_row = min(start_row + 100, total_rows)
        df_slice = df_selected.iloc[start_row:end_row]
        
        # 计算 Krippendorff's alpha
        data = df_slice.T.values.tolist()
        data_tuple = tuple(' '.join(map(str, row)) for row in data)
        newlistconvert = [[np.nan if (v == "*" or v == "N/A" or math.isnan(float(v))) else int(float(v)) for v in coder.split()] for coder in data_tuple]
        alpha = krippendorff.alpha(reliability_data=newlistconvert, level_of_measurement='nominal')
        alpha_list.append(alpha)
        print(f"{start_row + 1}-{end_row} rows, alpha:{alpha}")
    # 计算最终的平均值
    avg_alpha = sum(alpha_list) / len(alpha_list)
    print(f"nominal krippendorff alpha 3 huamn with model {model_name}: {round(avg_alpha, 3)}")
    # alpha = krippendorff.alpha(reliability_data=newlistconvert, level_of_measurement='interval')
    # print(f"interval krippendorff alpha 3 huamn with model {model_name} : {alpha}")

def main():
    df = pd.read_excel("CoSQA-plus/dataset/human-label/all_merge/human_query_code_pairs_1000_merge_all.xlsx")
    columns = "gpt4-turbo gpt4o gpt35 gpt351106 llama370binstruct claude3sonnet deepseek claude3haiku claude3opus Miking-exact Miking-50% Cookie-label-exact Cookie-label-50% thinker-exact thinker-50%"
    columns_list = columns.split()
    columns_half = "gpt4-turbo-50% gpt4o-50% gpt35-50% claude3sonnet-50% claude3opus-50% llama370binstruct-50%"
    columns_list_half = columns_half.split()
    # columns_list = ["claude3sonnet","claude3sonnet1","claude3sonnet2","claude3sonnet3"]
    columns_list3 = ["claude3sonnet-50%","claude3sonnet1-50%","claude3sonnet2-50%"]
    columns_list1 = ["Miking-exact","Cookie-label-exact","thinker-exact"]
    # columns_list1 = ["claude3sonnet","claude3haiku"]
    columns_list2 = ["Miking-50%","Cookie-label-50%","thinker-50%"]
    print("nominal krippendorff alpha 3 huamn(exact match)")
    for i in range(9):
        columns_list_temp = columns_list1.copy()
        columns_list_temp.append(columns_list[i])
        calculate_krippendorff_alpha(df,columns_list_temp,columns_list[i])
    print("nominal krippendorff alpha 3 huamn(50% match),notice:model still exact match")
    for i in range(6):
        columns_list_temp = columns_list2.copy()
        columns_list_temp.append(columns_list_half[i])
        calculate_krippendorff_alpha(df,columns_list_temp,columns_list_half[i])
    print("exact human just")
    calculate_krippendorff_alpha(df,columns_list1,"human only")
    print("50% human just")
    calculate_krippendorff_alpha(df,columns_list2,"human only")
    print("claude3 sonnet only")
    calculate_krippendorff_alpha(df,columns_list3,"claude3 sonnet only") 
if __name__ == "__main__":
    main()