import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ["TOKENIZERS_PARALLELISM"]="true"
import argparse
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import torch
import json
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModel
from model import Model
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


def convert_query_examples_to_features(js, tokenizer, args):
    """convert examples to token ids"""
    nl = (" ".join(js["doc"].split()))
    nl_tokens = tokenizer.tokenize(nl)[: args.nl_length - 4]
    nl_tokens = (
        [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token]
        + nl_tokens
        + [tokenizer.sep_token]
    )
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length
    return QueryInputFeatures(js["idx"], nl_tokens, nl_ids)


def convert_code_examples_to_features(js, tokenizer, args):
    """convert examples to token ids"""
    code = (" ".join(js["code"].split()))
    code_tokens = tokenizer.tokenize(code)[: args.code_length - 4]
    code_tokens = (
        [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token]
        + code_tokens
        + [tokenizer.sep_token]
    )
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length
    return CodeInputFeatures(js["idx"], code_tokens, code_ids)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class QueryDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
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
                convert_query_examples_to_features(js, tokenizer, args)
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
    def __init__(self, tokenizer, args, file_path=None, batch_size=5000):
        self.examples = []
        data = []
        # 读取文件
        with open(file_path) as f:
            if "json" in file_path:
                for js in json.load(f):
                    data.append(js)
        # 批量处理和并行化
        batched_data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        processed_batches = Parallel(n_jobs=-1)(delayed(self._process_batch)(batch, tokenizer, args) for batch in tqdm(batched_data, desc="Parallel processing/loading code"))
        self.examples = [example for batch in processed_batches for example in batch]
        # # 处理数据
        # for js in tqdm(data,desc="Processing/loading code"):
        #     self.examples.append(convert_code_examples_to_features(js, tokenizer, args))
        # 输出示例
        for idx, example in enumerate(self.examples[:1]):
            logger.info("*** Example ***")
            logger.info("idx: {}".format(idx))
            logger.info("code_tokens: {}".format([x.replace("\u0120", "_") for x in example.code_tokens]))

    def _process_batch(self, batch, tokenizer, args):
        return [convert_code_examples_to_features(js, tokenizer, args) for js in batch]
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].code_ids)



def get_similarity(args, model, tokenizer):
    query_file = args.query_file
    code_file = args.code_file
    query_dataset = QueryDataset(tokenizer, args, query_file)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(
        query_dataset,
        sampler=query_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
    )
    code_dataset = CodeDataset(tokenizer, args, code_file)
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
    similarity = []
    nl_vecs = []
    code_vecs = []
    for batch in tqdm(query_dataloader, desc="Processing queries"):  
        nl_inputs = batch.to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs)
            nl_vecs.append(nl_vec.cpu().numpy()) 

    for batch in tqdm(code_dataloader, desc="Processing code"):
        code_inputs = batch.to(args.device)    
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())
            
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    # 把code_vecs用pickle保存
    with open(, 'wb') as f:
        pickle.dump(code_vecs, f)
    scores = np.matmul(nl_vecs,code_vecs.T)
    
    logger.info("Obtain vectors and calculate similarity succeed!")
    return scores
    return similarity

def plot_similarity_distributions_per_query(similarity_matrix, num_bins=20):
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
        
        # 绘图设置
        plt.figure(figsize=(10, 6))
        plt.bar(bins[:-1], counts, width=(bins[1] - bins[0]), align='edge', edgecolor='black', 
               label=f'Similarity Intervals for Query {i+1}')
        plt.xlabel('Similarity Value')
        plt.ylabel('Number of Occurrences')
        plt.title(f'Distribution of Sorted Similarity Intervals for Query {i+1} (using {num_bins} bins)')
        plt.xticks(bins[:-1])
        plt.legend()
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        
        # 显示图形并保存到文件以避免立即关闭（可选）
        # plt.savefig(f'similarity_distribution_query_{i+1}.png')
        plt.show()

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
        default=256,
        type=int,
        help="Optional Code input sequence length after tokenization.",
    )
    parser.add_argument(
        "--eval_batch_size", default=4, type=int, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

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
    with open(args.query_file, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    all_similarity = []
    for model_name in args.model_name_or_path:
        logger.info("loading model %s", model_name)
        # load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = Model(model)
        logger.info("Parameters %s", args)
        model.to(args.device)
        # calculate similarity
        similarity = get_similarity(args,model,tokenizer)
        all_similarity.append(similarity)
        # 保存结果到csv文件
        modelname = model_name.replace("\\", "/").split("/")[-1]
        # df["similarity" + "_" + model_name] = similarity
        # df["labels" + "_" + model_name] = labels
        # output_file_name = args.data_file.replace(".json", ".csv")
        # df.to_csv(output_file_name, index=False)
    # df.to_csv(output_file_name, index=False)
    # 对每个query，计算每个code的平均相似度
    # all_similarity是一个三位的列表，all[i][j][k]代表第i个模型第j个query第k个code对应的相似度
    # 故求平均axis=0
    mean_similarity = np.mean(all_similarity, axis=0)
    plot_similarity_distributions_per_query(mean_similarity, 100)

if __name__ == "__main__":
    main()
