import json
import lucene
import argparse
import numpy as np
from tqdm import tqdm
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, TextField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
from java.nio.file import Paths


def CalculateMcRR(sort_list,eval_file,query_idx):
    with open(eval_file,'r') as f:
        data = json.load(f)
        
    # 找出给定query_idx的正确代码的code_idx
    code_idxs = [item['code-idx'] for item in data if item['query-idx'] == query_idx]
    # print(code_idxs)

    # 在list里面找到code_idx的rank并求倒数
    ranks = []
    inverse_ranks = []
    for code_idx in code_idxs:
        try:
            rank = sort_list.index(code_idx)+1
            if rank <= 1000:
                ranks.append(rank)
            else:
                ranks.append(0)
        #对于lucene,有可能在选出来的code_idx中是找不到某个正确答案的 
        except ValueError:
            ranks.append(0)
    ranks = sorted(ranks)
    i = 1
    for rank in ranks:
        if not rank==0:
            inverse_ranks.append(1/(rank-(i-1)))
            i+=1
        else:
            inverse_ranks.append(0)
    # print(f'ranks:{ranks}')    
     
    McRR = sum(inverse_ranks) / len(inverse_ranks)
    return McRR


def CalculateMMRR(sort_lists,eval_file,query_idxs):
    print(f'calculating MMRR--------------------')
    sum = 0
    cnt = 0
    for idx,item in tqdm(zip(query_idxs, sort_lists)):
        sum += CalculateMcRR(item,eval_file,idx)
        cnt += 1
        
    MMRR = sum / cnt 
    print(f'eval_mmrr:{MMRR}-------------------')
    return MMRR 

# sort_lists是按relevance降序排列的code_idxs
def CalculateMRR(sort_lists,eval_file,query_idxs):
    print(f'calculating MRR---------------------')
    with open(eval_file,'r') as f:
        data = json.load(f)
    ranks = []
    inverse_ranks = []
    for idx,item in tqdm(zip(query_idxs, sort_lists)):
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
        print(f'rank_i:{rank_i}')       
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
        
    
def main():
    parser = argparse.ArgumentParser()
    
    # pylucene没有训练过程，直接测试
    
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a json file).")
    parser.add_argument("--true_pairs_file", default=None, type=str,
                        help="A file contains all true pairs(a json file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional query input test data file to test the MMRR(a josn file).")
    
    args = parser.parse_args()
    
    # 初始化 JVM
    lucene.initVM()

    # 读取 codebase.json 文件
    with open(args.codebase_file, 'r') as f:
        codebase = json.load(f)

    # 创建索引存储目录
    index_path = Paths.get("/tmp/lucene_index")
    directory = MMapDirectory(index_path)

    # 创建标准分词器
    analyzer = StandardAnalyzer()

    # 配置索引写入器
    config = IndexWriterConfig(analyzer)
    writer = IndexWriter(directory, config)

    # 创建文档并添加到索引
    print(f'construct codebase----------')
    for item in tqdm(codebase):
        doc = Document()
        doc.add(TextField("code", item["code"], Field.Store.YES))
        doc.add(TextField("code-idx", str(item["code-idx"]), Field.Store.YES))
        writer.addDocument(doc)

    writer.close()

    # 读取 test.json 文件
    with open(args.test_data_file, 'r') as f:
        queries = json.load(f)

    # 搜索索引
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)

    # 配置查询解析器
    query_parser = QueryParser("code", analyzer)

    query_idxs = []
    sort_code_idxs = []
    error_idxs = []
    
    print(f'constructing queries--------------')
    # 计算每条查询与每条代码的相关性得分
    for query_item in tqdm(queries):
        query_idx = query_item["query-idx"]
        query_str = query_item["query"]
        
        try:
            # 解析查询
            query_parsed = query_parser.parse(query_str)

            hits = searcher.search(query_parsed, len(codebase)).scoreDocs  # 返回所有结果
            # 没有被选中的code不会出现在hits里面，分数会是零。

            code_idxs = []
            scores = []
            for hit in hits:
                doc = searcher.doc(hit.doc)
                code_idx = int(doc.get("code-idx"))
                code_idxs.append(code_idx)
                score = hit.score
                # print(f"query-idx:{query_idx} code-idx:{code_idx} score:{hit.score}")
                scores.append(score)

            query_idxs.append(query_item["query-idx"])
            # 对得分降序排列，并得到对应的code_idx的列表
            # 使用zip函数组合分数和索引
            combined = list(zip(scores, code_idxs))
            sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True) 

            # 可能会选出重复的，所以需要去重
            seen = set()
            sorted_combined_unique = [(a,b) for a,b in sorted_combined if not (b in seen or seen.add(b))]
            sort_code_idxs.append([idx for _, idx in sorted_combined_unique])    
        except BaseException:
            error_idxs.append(query_idx)
            continue
            




    reader.close()
    directory.close()

    mrr=CalculateMRR(sort_code_idxs,args.true_pairs_file,query_idxs)   
    mmrr=CalculateMMRR(sort_code_idxs,args.true_pairs_file,query_idxs)
    print(f'mrr={mrr}')
    print(f'mmrr={mmrr}')
    print(f'error_idxs:{error_idxs}---------------')
    
if __name__ == "__main__":
    main()
