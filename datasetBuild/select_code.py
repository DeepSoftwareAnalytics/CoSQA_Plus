import os

from sklearn.cluster import KMeans
os.environ["TOKENIZERS_PARALLELISM"]="true"
import argparse
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import torch
import json
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import RobertaConfig,RobertaTokenizer
from model import UniXcoderModel,CodeBertModel
import random
from sklearn.mixture import GaussianMixture
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from torch.nn import DataParallel
import pickle

logger = logging.getLogger(__name__)


class QueryInputFeatures(object):
    """A query sample feature"""

    def __init__(self, query_idx, nl_tokens, nl_ids):
        self.query_idx = query_idx
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids


class CodeInputFeatures(object):
    """A code sample feature"""

    def __init__(self, code_idx, code_tokens, code_ids):
        self.code_idx = code_idx
        self.code_tokens = code_tokens
        self.code_ids = code_ids


def convert_query_examples_to_features(js, tokenizer, args, model_name):
    """convert examples to token ids"""
    nl = (" ".join(js["doc"].split()))
    if model_name == "unixcoder-base":
        nl_tokens = tokenizer.tokenize(nl)[: args.nl_length - 4]
        nl_tokens = (
            [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token]
            + nl_tokens
            + [tokenizer.sep_token]
        )
        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = args.nl_length - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id]*padding_length
    elif model_name == "codebert-base":
        nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-2]
        nl_tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = args.nl_length - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id]*padding_length
    elif model_name == "codet5p-110m-embedding":
        nl_tokens = tokenizer.tokenize(nl)[: args.nl_length]
        nl_ids = tokenizer.encode(nl)[: args.nl_length]
        padding_length = args.nl_length - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id]*padding_length
    return QueryInputFeatures(js["idx"], nl_tokens, nl_ids)


def convert_code_examples_to_features(js, tokenizer, args, model_name):
    """convert examples to token ids"""
    code = (" ".join(js["code"].split()))
    if model_name == "unixcoder-base":
        code_tokens = tokenizer.tokenize(code)[: args.code_length - 4]
        code_tokens = (
            [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token]
            + code_tokens
            + [tokenizer.sep_token]
        )
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    elif model_name == "codebert-base":
        code_tokens = tokenizer.tokenize(code)[:args.code_length-2]
        code_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    elif model_name == "codet5p-110m-embedding":
        code_tokens = tokenizer.tokenize(code)[: args.code_length]
        code_ids = tokenizer.encode(code)[: args.code_length]
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    return CodeInputFeatures(js["idx"], code_tokens, code_ids)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class QueryDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None,model_name='unixcoder-base'):
        self.examples = []
        data = []
        # 读取文件
        with open(file_path) as f:
            if "json" in file_path:
                for js in json.load(f):
                    data.append(js)
        # 处理数据
        for js in tqdm(data,desc="Processing/loading query"):
            self.examples.append(
                convert_query_examples_to_features(js, tokenizer, args,model_name)
            )
        # 输出示例
        for idx, example in enumerate(self.examples[:1]):
            logger.info("*** Example ***")
            logger.info("idx: {}".format(idx))
            logger.info(
                "nl_tokens: {}".format(
                    [x.replace("\u0120", "_") for x in example.nl_tokens]
                )
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].nl_ids)
     


class CodeDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, batch_size=5000, model_name='unixcoder-base'):
        self.examples = []
        data = []
        # 读取文件
        with open(file_path) as f:
            if "json" in file_path:
                for js in json.load(f):
                    data.append(js)
        # 批量处理和并行化
        batched_data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        processed_batches = Parallel(n_jobs=-1)(delayed(self._process_batch)(batch, tokenizer, args, model_name) for batch in tqdm(batched_data, desc="Parallel processing/loading code"))
        self.examples = [example for batch in processed_batches for example in batch]
        # # 处理数据
        # for js in tqdm(data,desc="Processing/loading code"):
        #     self.examples.append(convert_code_examples_to_features(js, tokenizer, args))
        # 输出示例
        for idx, example in enumerate(self.examples[:1]):
            logger.info("*** Example ***")
            logger.info("idx: {}".format(idx))
            logger.info("code_tokens: {}".format([x.replace("\u0120", "_") for x in example.code_tokens]))

    def _process_batch(self, batch, tokenizer, args, model_name):
        return [convert_code_examples_to_features(js, tokenizer, args, model_name) for js in batch]
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].code_ids)



def get_embedding(args, model, tokenizer,model_name,code_dataset_name):
    query_file = args.query_file
    code_file = args.code_file
    query_dataset = QueryDataset(tokenizer, args, query_file,model_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(
        query_dataset,
        sampler=query_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
    )
    code_dataset = CodeDataset(tokenizer, args, code_file,5000,model_name)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(
        code_dataset,
        sampler=code_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4)
    # Eval!
    logger.info("***** Running get_similarity *****")
    logger.info("  Num query = %d", len(query_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    # obtain vector
    model.eval()
    nl_vecs = []
    code_vecs = []
    for batch1 in tqdm(query_dataloader, desc="Processing queries"):
            nl_inputs = batch1.to(args.device)
            with torch.no_grad():
                if model_name == 'unixcoder-base' or 'codebert-base':
                    nl_vec = model(nl_inputs)
                elif model_name == 'codet5p-110m-embedding':
                    nl_vec = model(nl_inputs)[0]
                nl_vecs.append(nl_vec.cpu().numpy())

    for batch in tqdm(code_dataloader, desc="Processing code"):
        code_inputs = batch.to(args.device)    
        with torch.no_grad():
            if model_name == 'unixcoder-base' or 'codebert-base':
                code_vec = model(code_inputs)
            elif model_name == 'codet5p-110m-embedding':
                code_vec = model(code_inputs)[0]
            code_vecs.append(code_vec.cpu().numpy())

    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    # 把code_vecs用pickle保存
    with open(f"{model_name}_{code_dataset_name}_code_vecs.pkl", 'wb') as f:
        pickle.dump(code_vecs, f)
    with open(f"{model_name}_nl_vecs.pkl", 'wb') as f:
        pickle.dump(nl_vecs, f)
    logger.info("Obtain and save vectors succeed!")
    # 计算相似度
    scores = np.matmul(nl_vecs,code_vecs.T)
    
    logger.info("Obtain vectors and calculate similarity succeed!")
    # return scores
    
def plot_similarity_distributions_per_query(similarity_matrix, model_name,num_bins=20):
    """
    为每个查询绘制一张相似度区间数量分布图。

    参数:
    similarity_matrix (numpy.ndarray): 多维数组，其形状为(n_queries, n_elements)，其中元素表示每个查询的相似度值。
    num_bins (int): 相似度分布图的区间数量，默认为10。

    返回:
    None，但会为每个查询显示一张相似度区间数量分布图。
    """
    # 确保输入是numpy数组且是二维的
    assert isinstance(similarity_matrix, np.ndarray) and len(similarity_matrix.shape) == 2, \
        "Input must be a 2D numpy ndarray"
    
    n_queries = similarity_matrix.shape[0]
    
    for i in range(n_queries):
        logger.info(f"Plotting distribution for query {i+1}")
        # 获取当前查询的相似度向量
        current_similarities = similarity_matrix[i]
        
        # 扁平化处理以便排序
        sorted_similarity = np.sort(current_similarities)
        
        # 计算区间边界
        bins = np.linspace(sorted_similarity.min(), sorted_similarity.max(), num_bins + 1)
        
        # 计算每个区间的值的数量
        counts, _ = np.histogram(sorted_similarity, bins=bins)
        # 输出最高数量区间对应的similarity值
        max_count_index = np.argmax(counts)
        max_count_similarity = bins[max_count_index]
        logger.info(f"{model_name}:Max count similarity for query {i+1}: {max_count_similarity}")
        # 绘图设置
        plt.figure(figsize=(10, 6))
        plt.bar(bins[:-1], counts, width=(bins[1] - bins[0]), align='edge', edgecolor='black', 
               label=f'Similarity Intervals for Query {i+1}')
        plt.xlabel('Similarity Value')
        plt.ylabel('Number of Occurrences')
        plt.title(f'Distribution of Sorted Similarity Intervals for Query {i+1} (using {num_bins} bins)')
        # 假设我们决定每5个bin显示一个标签
        selected_ticks = bins[:-1][::10]  # 选择每隔5个bin的位置
        plt.xticks(selected_ticks)
        # plt.xticks(bins[:-1])
        plt.legend()
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        
        # 显示图形并保存到文件以避免立即关闭（可选）
        plt.savefig(f'similarity_graph/{model_name}_similarity_distribution_query_{i+1}.png')
        # plt.show()

def select_code(args):
    # 读取每个模型对应的embedding结果
        all_code_vecs = []
        all_nl_vecs = []
        logger.info("Obtain code and query vector...")
        for model_name in args.model_name_or_path:
            with open("embedding/"+model_name+"_CSN_StaQC_code_vecs.pkl", 'rb') as f:
                all_code_vecs.append(pickle.load(f))
            with open("embedding/"+model_name+"_nl_vecs.pkl", 'rb') as f:
                all_nl_vecs.append(pickle.load(f))
            logger.info(f"Successfully obtain code and query vector from {model_name}")
        # all_code_vecs = np.array(all_code_vecs)
        # all_nl_vecs = np.array(all_nl_vecs)
        # query分批次获取相似度,选取top5的代码
        logger.info("Open query and code file...")
        with open(args.query_file,'r') as f:
            query_data = json.load(f)
        with open(args.code_file,'r') as f:
            code_data = json.load(f)
        query_num = len(query_data)
        logger.info(f"Successfully load {query_num} queries and {len(code_data)} codes")
        
        df = pd.DataFrame(query_data)
        df.set_index('idx', inplace=True)
        # 定义新列名以便存储top5代码数据
        new_columns = ['top1_code', 'top2_code', 'top3_code', 'top4_code', 'top5_code']
        
        # 初始化新列，稍后将填充实际数据
        for col in new_columns:
            df[col] = ''
        batch_size =204 # query num = 20604= 204 * 101 = 1717* 12
        # 遍历每个query
        logger.info("Start to calculate similarity and select code...")
        for i in tqdm(range(0, query_num, batch_size)):
            all_similarity = []
            total_similarity = None # 初始化总相似度矩阵，大小与一个模型的scores_normalized相同
            # 遍历每个模型
            for j in range(len(args.model_name_or_path)):
                logger.info(f"Calculating similarity for model {args.model_name_or_path[j]}...")
                code_vecs = all_code_vecs[j]
                nl_vecs = all_nl_vecs[j][i:i+batch_size]
                 # 当nl_vecs和code_vecs进行了L2归一化后，计算相似度时，直接计算内积即可
                scores = np.matmul(nl_vecs,code_vecs.T)
                # 对scores(cos similarity)进行最大绝对值归一化处理，以确保模型有相同的贡献权重
                max_abs_scores = np.max(np.abs(scores), axis=1, keepdims=True)
                scores_normalized = scores / max_abs_scores
                if total_similarity is None:
                    total_similarity = scores_normalized
                else:
                    total_similarity += scores_normalized
                # all_similarity.append(scores_normalized)
            # 对每个query，计算每个code的平均相似度
            # all_similarity是一个三位的列表，all[i][j][k]代表第i个模型第j个query第k个code对应的相似度
            # 故求平均axis=0
            logger.info("Calculating mean similarity for current batch...")
            # mean_similarity的计算很费时间，我想了很久，发现其实根本不用计算mean_similarity，直接用total_similarity便可
            # alright，the quicksort cost time
            # mean_similarity = np.mean(all_similarity, axis=0)
            # mean_similarity = total_similarity / len(args.model_name_or_path)
            top_n = 30
            sorted_idx = np.argpartition(-total_similarity, top_n, axis=1)[:, :30]
            # sorted_idx = np.argsort(total_similarity,axis=1, kind='quicksort', order=None)[:,::-1]
            logger.info(f"Processing query {i} ~ {i+batch_size}")
            for k in range(i,i+batch_size):
                if k >= query_num:
                    break
                # 获取当前query的代码信息
                query = query_data[k]
                query_idx = query['idx']
                # # 挑选前5个相似度最高的代码
                # temp_sorted_idx = sorted_idx[k-i][:5]
                # 选的代码的时候similarity从高到低一个一个选，只有不重复的代码才有机会被选到，选5个
                temp_sorted_idx = []
                temp_code_data = []
                temp_code_embedding = []
                for code_index in sorted_idx[k-i][:top_n]:
                    if len(temp_sorted_idx) >=5:
                        break
                    if code_data[code_index]['code'].strip() not in temp_code_data:
                        temp_code_data.append(code_data[code_index]['code'].strip())
                        temp_sorted_idx.append(code_index)
                        # temp_code_embedding.append(all_code_vecs[0][code_index]) # 这里all_code_vecs[j]代表选取第j个模型的向量来计算
                # # 用k-means聚类
                # kmeans = KMeans(n_clusters=5, random_state=0).fit(temp_code_embedding)
                # # 选取每个类别里面similarity最高的
                # select_code_index = []
                # for cluster_idx in range(5):
                #     select_code_index.append(temp_sorted_idx[np.argmax(kmeans.labels_==cluster_idx)])
                # temp_sorted_idx = select_code_index
                df.at[query_idx, 'top_code_index'] = str(temp_sorted_idx)
                for idx, rank in enumerate(range(1, 6)):  # 从1开始计数，对应top1到top5
                    code_idx = temp_sorted_idx[idx]
                    code_info =  code_data[code_idx] # 获取代码信息
                    if code_info:  # 确保代码索引存在
                        df.at[query_idx, f'top{rank}_code'] = str(code_info['code']).strip()
        # 保存结果
        # logger.info("Saving results to selected_code.csv...")
        # df.to_csv('selected_code1.csv')
        # df.to_excel('selected_code1.xlsx')
        if len(args.model_name_or_path) == 1:
            model_name = args.model_name_or_path[0]
            df.to_json(model_name+'_selected_code4.json',orient='records')
            df.to_pickle(model_name+'_selected_code4.pickle')
            logger.info("Task completed. Results saved.")
        else:
            df.to_json('selected_code4.json',orient='records')
            df.to_pickle('selected_code4.pickle')
            logger.info("Task completed. Results saved.")
        
                        
                    
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--query_file",
        default=None,
        type=str,
        help="query file to be calculated and divided.",
    )
    parser.add_argument(
        "--code_file",
        default=None,
        type=str,
        help="An optional input test data file to codebase (a jsonl file).",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        nargs="+",
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--nl_length",
        default=128,
        type=int,
        help="Optional NL input sequence length after tokenization.",
    )
    parser.add_argument(
        "--code_length",
        default=512,
        type=int,
        help="Optional Code input sequence length after tokenization.",
    )
    parser.add_argument(
        "--eval_batch_size", default=4, type=int, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument("--task",type=str,default='calculate similarity',help='',)
    # print arguments
    args = parser.parse_args()

    # set log
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)
    all_similarity = []
    if args.task == 'get_embedding':
        for model_name in args.model_name_or_path:
            logger.info("loading model %s", model_name)
            # load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name,trust_remote_code=True)
            if model_name == 'unixcoder-base':
                model = UniXcoderModel(model)
            elif model_name == 'codebert-base':
                model = CodeBertModel(model)
            logger.info("Parameters %s", args)
            model.to(args.device)
            # calculate similarity
            code_dataset_name = args.code_file.split("/")[-1].split(".")[0]
            get_embedding(args,model,tokenizer,model_name,code_dataset_name)
            # all_similarity.append(similarity)
            # 保存结果到csv文件
            # modelname = model_name.replace("\\", "/").split("/")[-1]
            # df["similarity" + "_" + model_name] = similarity
            # df["labels" + "_" + model_name] = labels
            # output_file_name = args.data_file.replace(".json", ".csv")
            # df.to_csv(output_file_name, index=False)
    elif args.task == 'select_code':
        select_code(args)
        
        
        # plot_similarity_distributions_per_query(mean_similarity,'all',100)
        # plot_similarity_distributions_per_query(scores,model_name, 100)
        
    # df.to_csv(output_file_name, index=False)
        

if __name__ == "__main__":
    main()
