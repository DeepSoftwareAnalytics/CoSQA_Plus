import json
from collections import Counter
from utils import excel_to_json
CSN99_EXCEL_FILE = "CoSQA-plus/dataset/CSN99/csn99.xlsx"
CSN99_JSON_FILE = "CoSQA-plus/dataset/CSN99/csn99.json"
def build_csn_test():
    '''
    这个函数用来构建CSN测试集
    '''
    excel_to_json(CSN99_EXCEL_FILE,CSN99_JSON_FILE)
    
    # excel_to_json("CoSQA-plus/dataset/CSN-high-relevance-python-test.xlsx","CoSQA-plus/dataset/CSN-high-relevance-python-test.json")
    # with open(f"CoSQA-plus/dataset/high-relevance-python-test.json","r") as f:
    #     code_pair_label = json.load(f)
    with open(f"CoSQA-plus/dataset/CSN99/csn99.json","r") as f:
        code_pair_label = json.load(f)
    # label_data_df = pd.read_csv("CoSQA-plus/dataset/dataset_annotation_GPT4_processed.csv")
    code_counter = Counter()
    query_counter = Counter()
    # 这个合并重复的code和query
    for item in code_pair_label:
        code_counter.update([item['code'].strip()])
        # 因为是测试用途，只保留label为1的pair的query
        if(item['label']==1):
            query_counter.update([item['query'].strip()])
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
    print("Query base build...")
    query_data = dict()
    query_base = []
    for idx,query in enumerate(query_counter.keys()):
        query_data[query] = {
            'query':query,
            'query-idx':idx,
        }
        query_base.append({
            'query-idx':idx,
            'query':query,
        })
    print("Query base build done.")
    # 构建最终的query-code-pairs
    print("Final query-code-pairs building...")
    final_query_code_pair_label = []
    pair_idx = 0
    for item in code_pair_label:
        if item['label']==1:
            final_query_code_pair_label.append({
                'pair-idx':pair_idx,
                'query-idx':query_data[item['query'].strip()]['query-idx'],
                'query':item['query'],
                'code-idx':code_data[item['code'].strip()]['code-idx'],
                'code':item['code'],
                'label':item['label'],
                # 'label':label_data_df[label_data_df['pair-index']==item['pair-index']]['judgement'].values[0],
            })
            pair_idx += 1 
    print("Final query-code-pairs building done.")
    print(f"codebase:{len(code_data)}")
    print(f'query base:{len(query_data)}')
    print(f"query-code-pairs:{len(final_query_code_pair_label)}")
    # 保存为json
    CSN99_PAIRS_JSON_FILE = "CoSQA-plus/dataset/CSN99/csn_99_query_code_pairs.json"
    CSN99_CODE_JSON_FILE = "CoSQA-plus/dataset/CSN99/csn_99_code_data.json"
    CSN99_QUERY_JSON_FILE = "CoSQA-plus/dataset/CSN99/csn_99_query_data.json"
    with open(CSN99_PAIRS_JSON_FILE,"w") as f:
        json.dump(final_query_code_pair_label,f)
        print(f"final_query_code_pairs.json saved: {len(final_query_code_pair_label)}")
    with open(CSN99_CODE_JSON_FILE,"w") as f:
        json.dump(code_base,f)
        print(f"codebase.json saved: {len(code_base)}")
    with open(CSN99_QUERY_JSON_FILE,"w") as f:
        json.dump(query_base,f)
        print(f"querybase.json saved: {len(query_base)}")
