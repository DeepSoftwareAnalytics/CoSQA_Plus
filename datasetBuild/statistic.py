import json
import pandas as pd
from process_data import get_claude_code
from collections import Counter
def query_statistic():
    files = ["CoSQA-plus/dataset/query.json","CoSQA-plus/dataset/cosqa-all.json"]
    with open(files[1],"r") as f:
        query_json = json.load(f)
        
    query_num = len(query_json)
    query_length = 0
    for item in query_json:
        query_length += len(item['doc'])
        
    print(f'query_num: {query_num}')
    print(f'query_avg_length: {query_length/query_num}')
def pairs_statistic():
    pairs_data = pd.read_csv("CoSQA-plus/dataset/stage_2_final/dataset_annotation_claude3sonnet_processed.csv")
    # 统计没有label为yes代码的query
    yes_label_query_data = pairs_data[pairs_data['judgement'] == "yes"]
    yes_label_query = []
    for idx,row in yes_label_query_data.iterrows():
        yes_label_query.append(row['query-index'])
    yes_label_query = set(yes_label_query)
    no_label_query = []
    for idx,row in pairs_data.iterrows():
        if row['query-index'] not in yes_label_query:
            no_label_query.append(row['query-index'])
    no_label_query = set(no_label_query)
    # get_claude_code(no_label_query)
    print(f"yes label query num:{len(yes_label_query)}")
    print(f"yes label query ratio:{len(yes_label_query)/len(pairs_data)}")
    print(f"no label query num:{len(no_label_query)}")
    print(f"no label query ratio:{len(no_label_query)/len(pairs_data)}")

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
    with open("CoSQA-plus/dataset/stage_3_augment/final_augment_codebase_label1.json","r") as f:
        augment_codebase_json = json.load(f)
    length = 0
    for item in augment_codebase_json:
        length += len(item['code'])
    print(f"avg_length: {length/len(augment_codebase_json)}")
        
def code_statistic():
    with open("CoSQA-plus/dataset/CSN-StaQC-code.json", "r") as f:
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

def augment_statistic(codebase_file,pairs_file):
    with open(codebase_file,"r") as f:
        augment_codebase_json = json.load(f)
    with open(pairs_file,"r") as f:
        augment_query_code_pairs = json.load(f)
    # 统计
    print(f"augment_codebase code num: {len(augment_codebase_json)}")
    print(f"augment_query_code_pairs num: {len(augment_query_code_pairs)}")
# pairs_statistic()
# selected_code_statistic()
# augment_statistic("CoSQA-plus/dataset/stage_3_augment/final_augment_codebase.json","CoSQA-plus/dataset/stage_3_augment/final_augment_query_code_pairs_for_search.json")
# augment_statistic("CoSQA-plus/dataset/stage_2_final/codebase.json","CoSQA-plus/dataset/stage_2_final/query_code_pairs_annotated.json")