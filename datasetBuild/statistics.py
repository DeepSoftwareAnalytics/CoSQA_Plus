# statistics.py
# Description: functions to calculate statistics about CoSQA+.
import json
import pandas as pd
from process_data import CSN_STAQC_CODE_FILE, divide_query_by_label, get_claude_code
from collections import Counter
from tabulate import tabulate
def query_statistic(query_file):
    """
    The function `query_statistic` calculates 
    the total number and the average length of the queries.
    """
    with open(query_file,"r", encoding="utf-8") as f:
        query_json = json.load(f)
        
    query_num = len(query_json)
    query_length = 0
    for item in query_json:
        query_length += len(item['doc'])
        
    print(f'query_num: {query_num}')
    print(f'query_avg_length: {query_length/query_num}')
    
def code_statistic(codebase_file):
    '''
    The function `code_statistic` calculates 
    the total number and the average length of the code snippets.
    '''
    with open(codebase_file,"r", encoding="utf-8") as f:
        codebase_json = json.load(f)
    code_num = len(codebase_json)
    code_length = 0
    for item in codebase_json:
        code_length += len(item['code'])
    print(f'code_num: {code_num}')
    print(f'code_avg_length: {code_length/code_num}')
    

    
def pairs_label_statistic(pairs_csv_file):
    '''
    The function `pairs_statistic` calculates 
    the number of queries with and without code labels.
    '''
    yes_label_query,no_label_query = divide_query_by_label(pairs_csv_file)
    len_yes_query, len_no_query = yes_label_query, no_label_query
    len_total_query = len_yes_query + len_no_query
    print(f"total pairs num:{len_total_query}")
    print(f"yes label query num:{len_yes_query}")
    print(f"yes label query ratio:{len_yes_query/len_total_query}")
    print(f"no label query num:{len_no_query}")
    print(f"no label query ratio:{len_no_query/len_total_query}")

def count_and_print_stats(data, source_name):
    total_num = 0
    total_length = 0
    more_length_counters = {1000: 0, 2000: 0, 5000: 0, 10000: 0}
    less_length_counters = {50: 0, 20: 0, 10: 0}
    for item in data:
        if item['source'] == source_name:
            total_num += 1
            total_length += len(item['code'])
            for threshold in more_length_counters.keys():
                if len(item['code']) >= threshold:
                    more_length_counters[threshold] += 1
                else:
                    break  # No need to check further once the condition is not met
            for threshold in less_length_counters.keys():
                if len(item['code']) > threshold:
                    break
                else:
                    less_length_counters[threshold] += 1
    
    avg_length = total_length / total_num if total_num else 0
    print(f'{source_name}: {total_num}')
    print(f'{source_name}_avg_length: {avg_length}')
    for length, count in more_length_counters.items():
        print(f'{source_name}_code length>={length}: {count}')
    for length, count in less_length_counters.items():
        print(f'{source_name}_code length<={length}: {count}')

def selected_code_statistic():
    with open("CoSQA-plus/dataset/stage_3_augment/final_augment_codebase_label1.json","r", encoding="utf-8") as f:
        augment_codebase_json = json.load(f)
    length = 0
    for item in augment_codebase_json:
        length += len(item['code'])
    print(f"avg_length: {length/len(augment_codebase_json)}")
        
def code_len_statistic():
    with open("CoSQA-plus/dataset/CSN-StaQC-code.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    sources = {'CSN': [], 'StaQC': []}
    for item in data:
        sources[item['source']].append(item)
    
    for source_name in sources:
        count_and_print_stats(sources[source_name], source_name)
    
    code_counter = Counter()
    for item in data:
        code_counter.update([item['code']])
    
    print(f'Total unique codes: {len(code_counter)}')
    print(f'Top 10 codes: {code_counter.most_common(10)}')

def select_code_statistic():
    # 统计检查
        with open("CoSQA-plus/dataset/selected_code3.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(CSN_STAQC_CODE_FILE,"r", encoding="utf-8") as f:
            code_data = json.load(f)
        count_20 = 0
        count_50 = 0
        count_2000 = 0
        count_5000 = 0
        code_counter = Counter()
        code_content_counter = Counter()
        for item in data:
            for i in range(1,6):
                if len(item[f'top{i}_code']) > 2000:
                    count_2000 += 1
                    # print(item['query'],":  ",item[f'top{i}_code'])
                    if len(item[f'top{i}_code']) > 5000:
                        count_5000 += 1
                if len(item[f'top{i}_code'])<50:
                    count_50 += 1
                    if(item[f'top{i}_code'] == "from __future__ import print_function"):
                        print(item['query'],":  ",item[f'top{i}_code'])
                    code_content_counter.update([item[f'top{i}_code'].strip()])
                    # print(item['query'],":  ",item[f'top{i}_code'])
                    if len(item[f'top{i}_code'])<20:
                        count_20 += 1
            code_index_list = item['top_code_index'][1:-1].strip().split(',')
            code_counter.update(code_index_list)
        # 统计重复出现的长度小于50的代码
        test_data = code_content_counter.most_common(10)
        print("Most Common Codes <50:")
        print(tabulate(test_data, headers=["Code", "Occurrence"], tablefmt="grid"))
        # 统计重复出现的代码
        table_data = code_counter.most_common(10)
        enriched_table_data = [
            (idx, count, code_data[int(idx)]['code'])  # 使用 get 防止 KeyError，若无对应片段则显示"N/A"
            for idx, count in table_data
        ]
        headers = ["Code Index", "Occurrence","Code Snippet"]
        print("Most Common Codes:")
        print(tabulate(enriched_table_data, headers=headers, tablefmt="grid"))
        
        # 统计来自CSN和StaQC的代码数量
        CSN_code_num = 0
        StaQC_code_num = 0
        code_length = 0 
        max_code_length = 0
        min_code_length = 5000
        for k,v in code_counter.items():
            k = int(k)
            code_length += len(code_data[k]['code'])
            if len(code_data[k]['code']) > max_code_length:
                max_code_length = len(code_data[k]['code'])
            if len(code_data[k]['code']) < min_code_length:
                min_code_length = len(code_data[k]['code'])
            if code_data[k]['source'] == 'CSN':
                CSN_code_num += 1
            elif code_data[k]['source'] == 'StaQC':
                StaQC_code_num += 1
        print("quert/item num:", len(data))
        print("total code num:",len(code_counter))
        print("CSN code num:",CSN_code_num)
        print("StaQC code num:",StaQC_code_num)
        print("Avg code length:",code_length/len(code_counter))
        print("Max code length:",max_code_length)
        print("Min code length:",min_code_length)
        # 注意下面输出的不同长度代码片段的统计是没有去重的
        print("code length > 2000:", count_2000)
        print("code length > 5000:", count_5000)
        print("code length < 20:", count_20)
        print("code length < 50:", count_50)



# file path
QUERY_FILE = "CoSQA-plus/dataset/query.json"
CODEBASE_FILE = "CoSQA-plus/dataset/codebase.json"
def main():
    query_statistic(QUERY_FILE)
    code_statistic(CODEBASE_FILE)

if __name__ == "__main__":
    main()

# pairs_statistic()
# selected_code_statistic()
# augment_statistic("CoSQA-plus/dataset/stage_3_augment/final_augment_codebase.json","CoSQA-plus/dataset/stage_3_augment/final_augment_query_code_pairs_for_search.json")
# augment_statistic("CoSQA-plus/dataset/stage_2_final/codebase.json","CoSQA-plus/dataset/stage_2_final/query_code_pairs_annotated.json")