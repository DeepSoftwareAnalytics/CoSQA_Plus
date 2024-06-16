
from collections import Counter
import json
import pickle
import random
from sklearn.cluster import KMeans

def divide():
    nl_embedding_file = "CoSQA-plus/dataset/text_embedding_large_query_embedding.pkl"
    # 读取原始数据集
    with open("CoSQA-plus/dataset/gpt4o_augment_query_code_pairs.json", "r") as f:
        pairs_data = json.load(f)
    with open("CoSQA-plus/dataset/query.json","r") as f:
        query_dataset = json.load(f)
    with open(nl_embedding_file, 'rb') as f:
        nl_vecs = pickle.load(f)
    # 排序字典的items()，然后提取values()
    sorted_nl_vecs = [nl_vecs[item['query-idx']] for item in query_dataset]
    # 进行聚类
    kmeans = KMeans(n_clusters=500, random_state=0).fit(sorted_nl_vecs)
    
def divide1():
    with open("CoSQA-plus/dataset/gpt4o_augment_query_code_pairs_for_search.json","r") as f:
        pairs_data = json.load(f)
    with open("CoSQA-plus/dataset/query.json","r") as f:
        query = json.load(f)
    query_counter = Counter()
    
    for item in pairs_data:
        query_counter.update([item['query-idx']])
    
    train_set = []
    dev_set = []
    test_set = []
    
    # 保留出现次数>=k的query
    for k in range(1,6):
        filtered_query_idx = []
        for query_idx,num in query_counter.items():
            if num == k:
                filtered_query_idx.append(query_idx)
        filtered_query = []
        for item in query:
            if item['query-idx'] in filtered_query_idx:
                filtered_query.append(item)
        # 打乱 filtered_query
        random.shuffle(filtered_query)
        print(f"filtered_query@{k}: {len(filtered_query_idx)}")

        # 划分数据集
        train_size = int(len(filtered_query) * 0.8)
        dev_size = int(len(filtered_query) * 0.1)
        test_size = len(filtered_query) - train_size - dev_size
        train_set.extend(filtered_query[:train_size])
        dev_set.extend(filtered_query[train_size:train_size+dev_size])
        test_set.extend(filtered_query[train_size+dev_size:])
    print(f"train_set: {len(train_set)}")
    print(f"dev_set: {len(dev_set)}")
    print(f"test_set: {len(test_set)}")
    # 划分pairs
    train_pairs = []
    dev_pairs = []
    test_pairs = []
    for item in pairs_data:
        if item['query-idx'] in [pair['query-idx'] for pair in train_set]:
            train_pairs.append(item)
        elif item['query-idx'] in [pair['query-idx'] for pair in dev_set]:
            dev_pairs.append(item)   
        elif item['query-idx'] in [pair['query-idx'] for pair in test_set]:
            test_pairs.append(item)
    print(f"train_pairs: {len(train_pairs)}")
    print(f"dev_pairs: {len(dev_pairs)}")
    print(f"test_pairs: {len(test_pairs)}")
    
    # 保存数据集
    
    with open("CoSQA-plus/dataset/train_query_code_pairs.json","w") as f:
        json.dump(train_pairs,f)
        
    with open("CoSQA-plus/dataset/dev_query_code_pairs.json","w") as f:
        json.dump(dev_pairs,f)
        
    with open("CoSQA-plus/dataset/test_query_code_pairs.json","w") as f:
        json.dump(test_pairs,f)
        
    with open("CoSQA-plus/dataset/train_query.json","w") as f:
        json.dump(train_set,f)
        
    with open("CoSQA-plus/dataset/dev_query.json","w") as f:
        json.dump(dev_set,f)
        
    with open("CoSQA-plus/dataset/test_query.json","w") as f:
        json.dump(test_set,f)
    
def trans_format():
    with open("CoSQA-plus/dataset/gpt4o_augment_codebase.json","r") as f:
        codebase = json.load(f)
    code_dict = dict()
    
    for item in codebase:
        code_dict[item['code']] = item['code-idx']
    # 保存为txt文件
    
    with open("CoSQA-plus/dataset/codebase.txt","w") as f:
        json.dump(code_dict,f)
    
    pre_process("CoSQA-plus/dataset/train_query_code_pairs.json","CoSQA-plus/dataset/train_query_code_pairs_processed.json")
    pre_process("CoSQA-plus/dataset/dev_query_code_pairs.json","CoSQA-plus/dataset/dev_query_code_pairs_processed.json")
    pre_process("CoSQA-plus/dataset/test_query_code_pairs.json","CoSQA-plus/dataset/test_query_code_pairs_processed.json")    
     
    pre_process()
        
def pre_process(input_file,output_file):
    with open(input_file,"r") as f:
        data = json.load(f)
    new_data = []
    for item in data:
        new_data.append({
            "idx":item['pair-idx'],
            "doc":item['query'],
            "code":item['code'],
            "code_tokens":"",
            "docstring_tokens":"",
            "label":item['label'],
            "retrieval_idx":item['code-idx']
        })
    with open(output_file,"w") as f:
        json.dump(new_data,f)
    
    
    
        
if __name__ == "__main__":
    divide1()
    