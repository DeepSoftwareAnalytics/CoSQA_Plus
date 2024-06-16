import sys 
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from tqdm import tqdm
logger = logging.getLogger(__name__)


def evaluate(args):
    logger.info("loading data...")
    with open(args.query_file, 'r') as f:
        query_dataset = json.load(f)
    with open(args.codebase_file,'r') as f:
        code_dataset = json.load(f)

    with open(args.code_embedding_file, 'rb') as f:
        code_vecs = pickle.load(f)
    with open(args.nl_embedding_file, 'rb') as f:
        nl_vecs = pickle.load(f)
    # 排序字典的items()，然后提取values()
    sorted_code_vecs = [code_vecs[item['code-idx']] for item in code_dataset]
    # sorted_code_vecs = [vec for idx, vec in sorted(code_vecs.items())]
    # sorted_nl_vecs = [vec for idx, vec in sorted(nl_vecs.items())]
    sorted_nl_vecs = [nl_vecs[item['query-idx']] for item in query_dataset]

    # 将排序后的嵌入列表转换为NumPy数组
    code_vecs_np = np.array(sorted_code_vecs)
    nl_vecs_np = np.array(sorted_nl_vecs)
    logger.info("embedding done and saved!")
    scores = np.matmul(nl_vecs_np,code_vecs_np.T)
    logger.info("scores done!")
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    logger.info("sort done!")
    # sort_ids不是idx，只是code投射下来的排名
    sort_idxs = []
    for sort_id in tqdm(sort_ids):
        sort_idx = []
        for i in sort_id[:1000]:
            sort_idx.append(code_dataset[i]['code-idx'])
        sort_idxs.append(sort_idx)

    
    
    # 要获取sort_ids对应的所有query-idxs
    query_idxs = []
    for example in tqdm(query_dataset):
        query_idxs.append(example['query-idx'])
        
    # 计算mmrr    
    # 不应该用传进来的file_name,这里要用所有正确的pair来评测（因为不止一条对的，传进来的file_name可能只有一条对的）
    logger.info("calculating mmrr...")
    mmrr = CalculateMMRR(sort_idxs,args.true_pairs_file,query_idxs)
    mrr = CalculateMRR(sort_idxs,args.true_pairs_file,query_idxs)

    result = {
        "eval_mrr":float(mrr),
        "eval_mmrr":float(mmrr)
    }

    return result

def CalculateMcRR(sort_list,data,query_idx):
  
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
        # 10000以后的置0
        try: 
            rank = sort_list.index(code_idx)+1
            if rank <= 1000:
                ranks.append(rank)
            else:
                ranks.append(0)
        except ValueError:
            ranks.append(0)
    ranks = sorted(ranks) #升序排序
    i=1
    for rank in ranks:
        if not rank==0:
            inverse_ranks.append(1/(rank-(i-1)))
            i+=1
        else:
            inverse_ranks.append(0)
    # print(f'ranks:{ranks}')
        
    MrRR = sum(inverse_ranks) / len(inverse_ranks)
    # print(f'The {query_idx}th query MrRR is {MrRR}')
    return MrRR

def CalculateMMRR(sort_lists,eval_file,query_idxs):
    sum = 0
    cnt = 0
    with open(eval_file,'r') as f:
        data = json.load(f)
    for idx,item in tqdm(zip(query_idxs, sort_lists),total=len(query_idxs)):
        sum += CalculateMcRR(item,data,idx)
        cnt += 1
        
    MMRR = sum / cnt 
    print(f'eval_mmrr:{MMRR}')
    return MMRR 

# sort_lists是按relevance降序排列的code_idxs
def CalculateMRR(sort_lists,eval_file,query_idxs):
    with open(eval_file,'r') as f:
        data = json.load(f)
    ranks = []
    inverse_ranks = []
    for idx,item in zip(query_idxs, sort_lists):
        # 找出给定query-idx的正确代码的code-idx的first one
        code_idxs = [item['code-idx'] for item in data if item['query-idx'] == idx] 
        # print(f'code_idxs:{code_idxs}')
        rank_i = []
        for code_idx in code_idxs:
            try:
                # 1000以后的置0
                rank = item.index(code_idx)+1
                if rank <= 1000:
                    rank_i.append(rank)
                else:
                    rank_i.append(0) 
            except ValueError:
                rank_i.append(0)
        # print(f'rank_i:{rank_i}')       
        # 只有0返回0，有0有正返回最小正整数
        rank_x = [num for num in rank_i if num > 0]
        rank_min = 0
        if rank_x:
            rank_min = min(rank_x)
        ranks.append(rank_min)
        # print(f'ranks:{ranks}')
    for rank in ranks:
        if not rank == 0:
            inverse_ranks.append(1/rank)
        else:
            inverse_ranks.append(0)
    MRR = sum(inverse_ranks) / len(inverse_ranks)
    print(f'eval_mrr:{MRR}')
    return MRR
     


def pre_process(args):
    if args.codebase_file_pre_process:
        filepath = args.codebase_file_pre_process
        with open(filepath,'r+') as f:
            data = json.load(f)
        for js in data:
            js["query-idx"] = ""
            js["query"] = ""
            js["pair-idx"] = ""
            js["label"] = ""
        with open(args.codebase_file,'w') as f:
            json.dump(data,f,indent=4)
        print(f'transformed!')
    
    if args.query_pre_process:
        filepath = args.query_pre_process
        with open(filepath,'r+') as f:
            data = json.load(f)
        for js in data:
            js["code-idx"] = ""
            js["code"] = ""
            js["pair-idx"] = ""
            js["label"] = ""
        with open(args.query_file,'w') as f:
            json.dump(data,f,indent=4)
        print(f'transformed!')
      

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MMRR(a json file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MMRR(a josn file).")
    # 添加新参数：处理前的codebase
    parser.add_argument("--codebase_file_pre_process", default=None, type=str,
                        help="Original codebase file(a json file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="Processed codebase file(a json file).") 
    # 添加新参数：处理前后的query
    parser.add_argument("--query_pre_process", default=None, type=str,
                       help="Original query file(a json file).")
    parser.add_argument("--query_file", default=None, type=str,
                       help="Processed query file(a json file).")
    
    # 添加新参数：所有正确的pairs
    parser.add_argument("--true_pairs_file", default=None, type=str,
                        help="A file contains all true pairs(a json file).")
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    # 添加新参数：code和nl的embedding文件路径
    parser.add_argument("--code_embedding_file", default=None, type=str,
                        help="The file contains the code embedding(a json file).")
    parser.add_argument("--nl_embedding_file", default=None, type=str,
                        help='The file contains the nl embedding(a json file).')
    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    # pre_process(args)
    if args.do_test:
        result = evaluate(args)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))


if __name__ == "__main__":
    main()
    