import json
import random
import pandas as pd
from collections import Counter
from tabulate import tabulate
from matplotlib import pyplot as plt
import pickle
import re
from tqdm import tqdm
def process_CSN():
    with open("CoSQA-plus/dataset/python_dedupe_definitions_v2.json","r") as f:
        data = json.load(f)
    new_data = []
    for i,item in enumerate(data):
        new_data.append(
            {
                'idx': "CSN-"+str(i),
                'code': item['function'],
                'url': item['url']
            }
        )
    with open("CoSQA-plus/dataset/CSN-code.json","w") as f:
        json.dump(new_data,f)
def merge_CSN_and_StaQC():
    with open("CoSQA-plus/dataset/CSN-code.json","r") as f:
        CSN_data = json.load(f)
    with open("CoSQA-plus/dataset/StaQC-code.json","r") as f:
        StaQC_data = json.load(f)
    new_data = []
    index = 0
    for i,item in enumerate(CSN_data):
        new_data.append(
            {
                'idx': index,
                'code': item['code'],
                'source': 'CSN',
            })
        index += 1
    for i,item in enumerate(StaQC_data):
        new_data.append(
            {
                'idx': index,
                'code': item['code'],
                'source': 'StaQC',
            })
        index += 1
    for i in range(3):
        print(f'处理完毕，第{i+1}个样例：{new_data[i]}')
    with open("CoSQA-plus/dataset/CSN-StaQC-code.json","w") as f:
        json.dump(new_data,f)
     

def process_query():
    with open("CoSQA-plus/dataset/cosqa-all.json","r") as f:
        data = json.load(f)
    new_data = []
    for i,item in enumerate(data):
        new_data.append(
            {
                'query-idx': i,
                'query': item['doc']
            }
        )
    with open("CoSQA-plus/dataset/query.json","w") as f:
        json.dump(new_data,f)
def remove_code_2000():
    with open("CoSQA-plus/dataset/StaQC-code.json","r") as f:
        code_json = json.load(f)
    
def get_claude_code(no_label_query):
    df = pd.read_excel("CoSQA-plus/dataset/stage_3_augment/claude3_code.xlsx")
    with open("CoSQA-plus/dataset/query.json") as f:
        query_json = json.load(f)
    # 筛选出没有label为yes的query对应的code,组成query-code pairs
    pairs_data = []
    for query_index in no_label_query:
        query = query_json[query_index]['query']
        pairs_data.append(
            {
                'pair-index': query_index,
                'query-index': query_index,
                'query': query,
                'code': df[df['query'] == query]['code'].values[0]
            }
        )
    print(f"claude3 query-code pairs num:{len(pairs_data)}")
    with open("CoSQA-plus/dataset/Claude3_query_code_pairs.json","w") as f:
        json.dump(pairs_data,f)
    
def select_1000_query():
    with open("CoSQA-plus/dataset/query.json","r") as f:
        query_json = json.load(f)
    # 随机挑选1000条
    select_query = random.sample(query_json,1000)
    with open("CoSQA-plus/dataset/query_1000.json","w") as f:
        json.dump(select_query,f)

def select_n_query(num):
    with open("CoSQA-plus/dataset/query.json","r") as f:
        query_json = json.load(f)
    # 随机挑选1000条
    select_query = random.sample(query_json,num)
    with open(f"CoSQA-plus/dataset/query_{num}.json","w") as f:
        json.dump(select_query,f)

def select_n_pairs(num):
    with open("CoSQA-plus/dataset/stage_2_final/final_query_code_pairs.json","r") as f:
        pairs = json.load(f)
        
    select_pairs = random.sample(pairs,num)
    with open(f"CoSQA-plus/dataset/human_query_code_pairs_{num}.json","w") as f:
        json.dump(select_pairs,f)
        
def select_pairs_1000_query():
    """
     随机筛选出1000条query并调用from_top5_to_individual
    """
    models = ['unixcoder-base','codebert-base','codet5p-110m-embedding']
    with open("CoSQA-plus/dataset/query_1000.json","r") as f:
        query_json = json.load(f)
    for model in models:
        with open(f"{model}_selected_code4.json","r") as f:
            pairs_json = json.load(f)
        # 筛选出1000条query对应的pairs
        pairs_json = [item for item in pairs_json if item['query'] in [item['query'] for item in query_json]]
        with open(f"CoSQA-plus/dataset/{model}_selected_code_1000.json","w") as f:
            json.dump(pairs_json,f)
        print(f"{model} selected query num: {len(pairs_json)}")
        # 转变形式
        from_top5_to_individual(model)
    

def merge_query_code_pair():
    """
    将query和code的pair合并
    query-idx和code-idx相同则合并
    生成这个是为了打标签
    """
    models = ['unixcoder-base','codebert-base','codet5p-110m-embedding']
    all_pairs_json = []
    # 汇集合并每个模型的query-code pairs
    for model in models:
        with open(f"CoSQA-plus/dataset/{model}_query_code_pair_label_1000.json","r") as f:
            pairs_json = json.load(f)
        for item in pairs_json:
            new_item = {
                'query-idx': item['query-index'],
                'query': item['query'].strip(),
                'code': item['code'].strip(),
                'code-idx': item['code-index']
            }
            if new_item not in all_pairs_json:
                all_pairs_json.append(new_item)
    print(f"merge pairs num before remove: {len(all_pairs_json)}")
    # 在origin_pairs中存在的pairs也排除掉
    with open("CoSQA-plus/dataset/query_code_pair_label.json","r") as f:
        origin_pairs_json = json.load(f)
    for item in origin_pairs_json:
        new_item = {
            'query-idx': item['query-index'],
            'query': item['query'].strip(),
            'code': item['code'].strip(),
            'code-idx': item['code-index']
        }
        if new_item in all_pairs_json:
            print(f"remove {new_item}")
            all_pairs_json.remove(new_item)
    print(f"merge pairs num after remove: {len(all_pairs_json)}")
    all_pairs_json_with_idx = []
    for idx,item in enumerate(all_pairs_json):
        all_pairs_json_with_idx.append({
            'pair-index': idx,
            'query-index': item['query-idx'],
            'query': item['query'],
            'code': item['code'],
            'code-index': item['code-idx']
        })
    
    with open("CoSQA-plus/dataset/1000_query_code_pair_merge.json","w") as f:
        json.dump(all_pairs_json_with_idx,f)

def build_augment_codebase_and_pairs():
    """
    融合origin_pairs和claude_pairs
    并将新的code加入codebase形成augment_codebase
    """
    with open("CoSQA-plus/dataset/stage_2_final/query_code_pairs_annotated.json","r") as f:
        origin_pairs_json = json.load(f)
    with open("CoSQA-plus/dataset/stage_3_augment/Claude3_query_code_pairs.json","r") as f:
        claude_pairs_json = json.load(f)
    with open("CoSQA-plus/dataset/stage_2_final/codebase.json","r") as f:
        origin_codebase_json = json.load(f)
    # build augment codebase
    print(f"build augment codebase...")
    augment_codebase_json = origin_codebase_json
    claude_code_dict = dict()
    code_idx = len(origin_codebase_json)
    print(f"build origin codebase done...len:{len(augment_codebase_json)}")
    for item in claude_pairs_json:
        augment_codebase_json.append({
            'code-idx': code_idx,
            'code': item['code']
        })
        claude_code_dict[item['code']] = code_idx
        code_idx+=1
    print(f"build augment codebase done...len:{len(augment_codebase_json)}")
    with open("CoSQA-plus/dataset/stage_3_augment/final_augment_codebase.json","w") as f:
        json.dump(augment_codebase_json,f)
    # build augment pairs
    print(f"build augment pairs...")
    augment_pairs_json = []
    for item in origin_pairs_json:
        augment_pairs_json.append({
            'pair-idx': item['pair-idx'],
            'query-idx': item['query-idx'],
            'query': item['query'],
            'code-idx': item['code-idx'],
            'code': item['code'],
            'label': item['label']
            })
    pair_idx = len(origin_pairs_json)
    for item in claude_pairs_json:
        augment_pairs_json.append({
            'pair-idx': pair_idx,
            'query-idx': item['query-index'],
            'query': item['query'],
            'code-idx': claude_code_dict[item['code']],
            'code': item['code'],
            'label':1
        })
        pair_idx+=1
    print(f"build augment pairs done...len:{len(augment_pairs_json)}")
    with open("CoSQA-plus/dataset/stage_3_augment/final_augment_query_code_pairs.json","w") as f:
        json.dump(augment_pairs_json,f)
        
def select_code_statistic():
    # 统计检查
        with open("CoSQA-plus/dataset/selected_code3.json", "r") as f:
            data = json.load(f)
        with open("CoSQA-plus/dataset/CSN-StaQC-code.json","r") as f:
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

def df_to_json():
    df = pd.read_pickle("CoSQA-plus/dataset/selected_code2.pickle")
    df.to_json("CoSQA-plus/dataset/selected_code2.json",orient="records")

def json_to_csv(json_file,df_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.to_csv(df_file,index=False)
def from_top5_to_individual(model_name=''):
    with open(f"CoSQA-plus/dataset/multi-model/{model_name}_selected_code.json","r") as f:
        top5_code_data = json.load(f)
    with open("CoSQA-plus/dataset/stage_1_origin/CSN-StaQC-code.json","r") as f:
        all_code_data = json.load(f)
    code_data = []
    pair_index = 0
    for idx,item in enumerate(top5_code_data):
        code_index_list = item['top_code_index'][1:-1].strip().split(',')
        for code_index in code_index_list:
            code_index = int(code_index)
            code_data.append({
                'pair-index':pair_index,
                'query-index':idx,
                'query':item['query'],
                'code-index':code_index,
                'code':all_code_data[code_index]['code'],
            })
            pair_index+=1
    print(len(code_data))
    with open(f"CoSQA-plus/dataset/multi-model/{model_name}_query_code_pair_label.json","w") as f:
        json.dump(code_data,f)

def build_codebase(model_name=''):
    with open(f"CoSQA-plus/dataset/multi-model/{model_name}_query_code_pair_label.json","r") as f:
        code_pair_label = json.load(f)
    # label_data_df = pd.read_csv("CoSQA-plus/dataset/dataset_annotation_GPT4_processed.csv")
    code_counter = Counter()
    # 这个合并重复的code
    for item in code_pair_label:
        code_counter.update([item['code'].strip()])
    # 遍历code_counter,构建codebase
    print("Codebase building...")
    code_data = dict()
    code_base = []
    for idx,code in enumerate(code_counter.keys()):
        code_data[code] = {
            'code':code,
            'code-idx':idx,
        }
        code_base.append({
            'code-idx':idx,
            'code':code,
        })
    print("Codebase building done.")
    # 构建最终的query-code-pairs
    print("Final query-code-pairs building...")
    final_query_code_pair_label = []
    for item in code_pair_label:
        final_query_code_pair_label.append({
            'pair-idx':item['pair-index'],
            'query-idx':item['query-index'],
            'query':item['query'],
            'code-idx':code_data[item['code'].strip()]['code-idx'],
            'code':item['code'],
            # 'label':label_data_df[label_data_df['pair-index']==item['pair-index']]['judgement'].values[0],
        })
    print("Final query-code-pairs building done.")
    print(f"codebase:{len(code_data)}")
    print(f"query-code-pairs:{len(final_query_code_pair_label)}")
    # 保存为json
    with open(f"CoSQA-plus/dataset/multi-model/{model_name}_final_query_code_pairs.json","w") as f:
        json.dump(final_query_code_pair_label,f)
        print(f"final_query_code_pairs.json saved: {len(final_query_code_pair_label)}")
    with open(f"CoSQA-plus/dataset/multi-model/{model_name}_code_data.json","w") as f:
        json.dump(code_base,f)
        print(f"codebase.json saved: {len(code_base)}")

def multi_model_process():
    models = ['unixcoder-base','codebert-base','codet5p-110m-embedding']
    for model_name in models:
        print(f"Processing {model_name}...")
        from_top5_to_individual(model_name)
        build_codebase(model_name)
        print(f"Processing {model_name} done.")
def divide_dataset():
    with open("CoSQA-plus/dataset/stage_3_augment/final_augment_query_code_pairs.json","r") as f:
        augment_query_code_pairs = json.load(f)
    augment_query_code_pairs_for_search = []
    for item in augment_query_code_pairs:
        if item['label'] == 1:
            augment_query_code_pairs_for_search.append(item)
    with open("CoSQA-plus/dataset/stage_3_augment/final_augment_query_code_pairs_for_search.json","w") as f:
        json.dump(augment_query_code_pairs_for_search,f)

def judgement_extraction(input_file, processed_file):
    df = pd.read_csv(input_file, index_col=0)
    # 从answer列中origin_answer提取
    # 使用正则表达式匹配 "judgement": "([^"]+)"
    count_yes = 0
    count_no = 0
    for index, row in df.iterrows():
        data_str = row['origin_answer']
        match = re.search(r'("judgement"|"judgment"): "([^"]+)"', data_str)
        if match:
            judgement_value = match.group(2)
            df.at[index, 'judgement'] = judgement_value.lower()
            if(judgement_value == "yes") : 
                count_yes += 1
            elif(judgement_value == "no") : 
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
def judgement_extraction1(input_file1,processed_file1):
    df = pd.read_csv(input_file1, index_col=0)
    # 从answer列中origin_answer提取
    # 使用正则表达式匹配 "judgement": "([^"]+)"
    for index, row in df.iterrows():
        data_str = str(row['judgement'])
        if "yes" in data_str and "no" in data_str:
            df.at[index, 'final_judgement'] = ""
        elif "yes" in data_str:
            df.at[index, 'final_judgement'] = "yes"
        elif "no" in data_str:
            df.at[index, 'final_judgement'] = "no"
        else:
            df.at[index, 'final_judgement'] = ""
            print(f"No match found for pair-index {index}")
    df.to_csv(processed_file1)
def remove_no_answer_row(input_file, processed_file):
    df = pd.read_csv(input_file, index_col=0)
    # 从answer列中origin_answer提取
    # 使用正则表达式匹配 "judgement": "([^"]+)"
    remove_index = []
    for index, row in df.iterrows():
        data_str = row['origin_answer']
        match = re.search(r'("judgement"|"judgment"): "([^"]+)"', data_str)
        if match:
            pass
            # print(f"Match found for pair-index {index}")
        else:
            remove_index.append(index)
            print(f"No match found for pair-index {index}")
    df.drop(remove_index, inplace=True)
    df.to_csv(processed_file)
    print(f"processed file saved: {processed_file} len:{len(df)}")
def remove_empty_row(input_file, processed_file):
    df = pd.read_csv(input_file, index_col=0)
    df = df.dropna(subset=['origin_answer'])
    df.to_csv(processed_file)
    print(f'processed file saved: {processed_file} len:{len(df)}')

def drop_n_pair_index(input_file,processed_file,pair_index):
    df = pd.read_csv(input_file, index_col=0)
    df = df.drop(df[df['pair-index'] == pair_index].index)
    df.to_csv(processed_file)
    print(f'processed file saved: {processed_file} len:{len(df)}')
def change_to_no(input_file,processed_file,pair_index):
    df = pd.read_csv(input_file, index_col=0)
    df.at[pair_index, 'judgement'] = "no"
    df.to_csv(processed_file)
    print(f'processed file saved: {processed_file} len:{len(df)}')
def filter_codebase():
    with open("CoSQA-plus/dataset/stage_3_augment/final_augment_codebase.json","r") as f:
        codebase = json.load(f)
    
    with open("CoSQA-plus/dataset/stage_3_augment/final_augment_query_code_pairs.json","r") as f:
        query_code_pairs = json.load(f)
    new_codebase = []
    # 只保留在query_code_pairs中出现的code
    for item in tqdm(codebase):
        for pair in query_code_pairs:
            if pair['label']==1 and item['code-idx']==pair['code-idx']:
                new_codebase.append(item)
                break
    print(f"new codebase len:{len(new_codebase)}")
    with open("CoSQA-plus/dataset/stage_3_augment/final_augment_codebase_label1.json","w") as f:
        json.dump(new_codebase,f)

def merge_label():
    # models = ['gpt4o', 'gpt35', 'gpt351106']
    # models = ['llama370binstruct','claude3sonnet']
    # models = ['claude3opus']
    models = ['claude3sonnet1','claude3sonnet2','claude3sonnet3']

    # 加载原始的query_code_pairs
    with open("CoSQA-plus/dataset/human-label/human_query_code_pairs_1000.json", "r") as f:
        query_code_pairs = json.load(f)

    # 遍历每个模型
    for model in models:
        judgement_extraction(f"CoSQA-plus/dataset/human-label/human_query_code_pairs_1000_{model}.csv",f"CoSQA-plus/dataset/human-label/human_query_code_pairs_1000_{model}_processed.csv")
        
        print(f"model: {model}")
        
        # 加载模型的标签数据
        df = pd.read_csv(f"CoSQA-plus/dataset/human-label/human_query_code_pairs_1000_{model}_processed.csv")
        
        # 遍历query_code_pairs列表，并修改每个字典
        for index, pairs in enumerate(query_code_pairs):
            # 在df中查找与当前pairs['pair-index']匹配的行
            matching_rows = df[df['pair-index'] == pairs['pair-idx']]
            
            # 如果找到了匹配的行，则添加新的键值对到pairs字典中
            if not matching_rows.empty:
                judgement = matching_rows['judgement'].values[0]
                query_code_pairs[index][model] = 1 if judgement == "yes" else 0
    print("start to save...")          
    with open("CoSQA-plus/dataset/human-label/human_query_code_pairs_1000_merge5.json","w") as f:
        json.dump(query_code_pairs,f)
    json_to_csv("CoSQA-plus/dataset/human-label/human_query_code_pairs_1000_merge5.json","CoSQA-plus/dataset/human-label/human_query_code_pairs_1000_merge5.csv")

def relabel_cosqa():
    with open("CoSQA-plus/dataset/CoSQA/cosqa-retrieval-train-19604.json","r") as f:
        cosqa = json.load(f)
    new_label = pd.read_excel("CoSQA-plus/dataset/CoSQA/cosqa_all_gpt4_match_zero_shot_cot_processed1.xlsx")

    for index,item in enumerate(cosqa):
        new_judgement = new_label.loc[new_label['idx']==item['idx'],'judgement'].values[0]
        if new_judgement == "yes":
            cosqa[index]['label'] = 1
        elif new_judgement == "no":
            cosqa[index]['label'] = 0
    with open("CoSQA-plus/dataset/CoSQA/cosqa-retrieval-train-19604_new.json","w") as f:
        json.dump(cosqa,f)

def total_pairs_build():
    print("start to load...")
    df = pd.read_csv("CoSQA-plus/dataset/stage_2_final/dataset_annotation_claude3sonnet_processed.csv")
    codebase = pd.read_json("CoSQA-plus/dataset/stage_2_final/codebase.json")
    query = pd.read_json("CoSQA-plus/dataset/query.json")
    # with open("CoSQA-plus/dataset/stage_2_final/codebase.json","r") as f:
    #     codebase = json.load(f)
    # with open("CoSQA-plus/dataset/query.json","r") as f:
    #     query = json.load(f)
    query_code_pairs = []
    for idx, item in tqdm(df.iterrows(), total=len(df)):
        judgement = item['judgement']
        # 将query和codebase中的相应行转换为字典
        query_row = query[query['query-idx'] == item['query-index']].to_dict(orient='records')
        code_row = codebase[codebase['code-idx'] == item['code-index']].to_dict(orient='records')
        
        # 确保每个查询和代码行都恰好有一条记录
        if len(query_row) == 1 and len(code_row) == 1:
            query_code_pairs.append({
                "pair-idx": item['pair-index'],
                "query-idx": item['query-index'],
                "query": query_row[0]['query'],
                "code-idx": item['code-index'],
                "code": code_row[0]['code'],
                "label": 1 if judgement == "yes" else 0,
            })
        else:
            print(f"Warning: Incorrect number of records found for query_idx {item['query-index']} or code_idx {item['code-index']}.")
    with open("CoSQA-plus/dataset/stage_2_final/query_code_pairs_annotated.json","w") as f:
        json.dump(query_code_pairs,f)

def cut_code_embedding_pkl():
    with open("CoSQA-plus/dataset/text_embedding_large_code_embedding.pkl","rb") as f:
        code_embedding = pickle.load(f)
    with open("CoSQA-plus/dataset/stage_2_final/codebase.json","r") as f:
        codebase = json.load(f)
    cut_length = len(codebase)
    code_embedding_copy = code_embedding.copy()
    for key in code_embedding_copy.keys():
        if key >= cut_length:
            del code_embedding[key]
    print(f"cut code embedding length: {len(code_embedding)}")
    with open("CoSQA-plus/dataset/code_embedding.pkl","wb") as f:
        pickle.dump(code_embedding,f)

# cut_code_embedding_pkl()
# build_augment_codebase_and_pairs()
# relabel_cosqa()
# total_pairs_build()
# change_to_no("CoSQA-plus/dataset/stage_2_final/dataset_annotation_claude3sonnet_processed.csv","CoSQA-plus/dataset/stage_2_final/dataset_annotation_claude3sonnet_processed.csv",45350)
# drop_n_pair_index("CoSQA-plus/dataset/stage_2_final/dataset_annotation_claude3sonnet.csv","CoSQA-plus/dataset/stage_2_final/dataset_annotation_claude3sonnet.csv",45350)
# remove_no_answer_row("CoSQA-plus/dataset/stage_2_final/dataset_annotation_claude3sonnet.csv","CoSQA-plus/dataset/stage_2_final/dataset_annotation_claude3sonnet.csv")
# judgement_extraction("CoSQA-plus/dataset/stage_2_final/dataset_annotation_claude3sonnet.csv","CoSQA-plus/dataset/stage_2_final/dataset_annotation_claude3sonnet_processed.csv")
# merge_label()
# select_n_pairs(1000)
# filter_codebase()
# multi_model_process()
# judgement_extraction("CoSQA-plus/dataset/multi-model/dataset_annotation_1000_query_GPT4.csv","CoSQA-plus/dataset/multi-model/dataset_annotation_1000_query_GPT4_processed.csv")
# build_augment_
# codebase_and_pairs()
# divide_dataset()
# pairs_statistic()
# select_1000_query()
# select_pairs_1000_query()
# merge_query_code_pair()
# build_codebase()
# process_query()
# merge_CSN_and_StaQC()
# statistic()
# query_statistic()
# code_statistic()
# df_to_json()
# select_code_statistic()
# from_top5_to_individual()
# json_to_csv("CoSQA-plus/dataset/query_code_pair_label.json","CoSQA-plus/dataset/query_code_pair_label.csv")
# json_to_csv("CoSQA-plus/dataset/query_code_pair.json","CoSQA-plus/dataset/query_code_pair.csv")
# judgement_extraction("dataset_annotation_GPT4.csv","CoSQA-plus/dataset/dataset_annotation_GPT4_processed.csv")
# judgement_extraction1("CoSQA-plus/dataset/dataset_annotation_GPT4_processed.csv","CoSQA-plus/dataset/dataset_annotation_GPT4_processed1.csv")
# remove_empty_row("dataset_annotation_GPT4.csv","dataset_annotation_GPT4.csv")
# remove_no_answer_row("dataset_annotation_GPT4.csv","dataset_annotation_GPT4.csv")
# drop_5098("dataset_annotation_GPT4.csv","dataset_annotation_GPT4.csv")
