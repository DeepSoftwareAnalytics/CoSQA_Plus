import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import logging
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed  # 引入joblib进行并行处理
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 示例代码和查询
query = 'CoSQA-plus/dataset/query.json'
codebase = 'CoSQA-plus/dataset/augment_codebase_label1.json'
codebase2 = 'CoSQA-plus/dataset/augment_codebase.json'
eval_file = 'CoSQA-plus/dataset/augment_query_code_pairs_for_search.json'

def CalculateMrRR(sort_list,data,query_idx):
    # 找出给定query-idx的正确代码的code-idx
    code_idxs = [item['code-idx'] for item in data if item['query-idx'] == query_idx]
    # print(code_idxs)
    # 要给code_idxs升序排序（不然会出现分母为零的情况）
    code_idxs = sorted(code_idxs)
    # print(code_idxs)
    
    # 在list里面找到code-idx的rank并求倒数
    ranks = []
    inverse_ranks = []
    
    for code_idx in code_idxs:
        ranks.append(sort_list.index(code_idx)+1)
            # inverse_ranks.append(1/(sort_list.index(code_idx) + 2 - i))

        # 对于lucene,有可能在选出来的code-idx中是找不到某个正确答案的 
        # except ValueError:
        #     inverse_ranks.append(0)
    # 改了之后对于lucene的思路是不一样的，测lucene的时候要另外改
    ranks = sorted(ranks) #升序排序
    i=1
    for rank in ranks:
        inverse_ranks.append(1/(rank-(i-1))) 
        i+=1
    # print(f'ranks:{ranks}')
        
    MrRR = sum(inverse_ranks) / len(inverse_ranks)
    # print(f'The {query_idx}th query MrRR is {MrRR}')
    return MrRR

def CalculateMMRR(sort_lists, eval_file, query_idxs):
    with open(eval_file, 'r') as f:
        data = json.load(f)
    
    mmrr_values = [CalculateMrRR(sort_list, data, idx) for sort_list, idx in tqdm(zip(sort_lists, query_idxs))]
    MMRR = np.mean(mmrr_values)
    
    return MMRR

def main():
    logging.info("Loading data...")
    with open(query, 'r') as q:
        data_query = json.load(q)

    with open(codebase, 'r') as c:
        data_code = json.load(c)
        
    with open(codebase2, 'r') as c2:
        data_code2 = json.load(c2) 

    # 创建一个CountVectorizer对象
    vectorizer = CountVectorizer(max_features=2000)
    logging.info("Calculating similarities and sorting...")

    # 将所有代码段预先向量化
    code_texts = [c['code'] for c in data_code]
    code_vectors = vectorizer.fit_transform(code_texts).toarray()
    # 预先计算查询向量
    query_vectors = [vectorizer.transform([q['query']]).toarray()[0] for q in data_query]
    logging.info("Calculating similarity matrix...")
    # 假设query_vectors和code_vectors是经过向量化后的稀疏矩阵
    query_matrix = csr_matrix(query_vectors)
    code_matrix = csr_matrix(code_vectors)
    # 计算点乘
    dot_product = query_matrix.dot(code_matrix.T)
    # 计算查询向量和代码向量的范数
    query_norms = np.sqrt(query_matrix.power(2).sum(axis=1)).A1  # 转换为一维数组
    code_norms = np.sqrt(code_matrix.power(2).sum(axis=1)).A1
    # 归一化得到余弦相似度矩阵
    denom = query_norms[:, np.newaxis] * code_norms
    cos_sim_matrix = dot_product.toarray() / denom

    # 注意：确保处理可能出现的除以零的情况，即分母为0时，可能需要将这些值设置为0或其他合理值。
    similarities = np.where(denom != 0, cos_sim_matrix, 0)
    sort_ids = np.argsort(similarities, axis=-1, kind='quicksort', order=None)[:,::-1]    
    logging.info("Sorting complete.")
    # sort_ids不是idx，只是code投射下来的排名
    
    sort_idxs = []
    for sort_id in tqdm(sort_ids):
        sort_idx = []
        for i in sort_id:
            sort_idx.append(data_code[i]['code-idx'])
        sort_idxs.append(sort_idx)

    # 要获取sort_ids对应的所有query-idxs
    query_idxs = []
    for example in tqdm(data_query):
        query_idxs.append(example['query-idx'])
    logging.info("Evaluation...")
    MMRR_result = CalculateMMRR(sort_idxs, eval_file, query_idxs)
    print(f'eval_mmrr: {MMRR_result}')
    print(f'bow label1+label0')

if __name__ == '__main__':
    main()