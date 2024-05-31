import argparse
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import torch
import json
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModel
from model import Model
import random
import os
from sklearn.mixture import GaussianMixture
import pandas as pd
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url


def convert_examples_to_features(js, tokenizer, args):
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length - 4]
    code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 4]
    nl_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js['url'] if "url" in js else js["retrieval_idx"])

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        # 读取文件
        with open(file_path) as f:
            if "json" in file_path:
                for js in json.load(f):
                    data.append(js) 
        # 处理数据
        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))
        # 输出示例
        for idx, example in enumerate(self.examples[:1]):
            logger.info("*** Example ***")
            logger.info("idx: {}".format(idx))
            logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
            logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
            logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
            logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids))

def get_similarity(args, model, tokenizer, file_name):
    query_dataset = TextDataset(tokenizer, args, file_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Eval!
    logger.info("***** Running get_similarity *****")
    logger.info("  Num query-code pairs = %d", len(query_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    # obtain vector
    model.eval()
    similarity= []
    for batch in query_dataloader:
        code_inputs = batch[0].to(args.device)
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs)
            code_vec = model(code_inputs=code_inputs)
            # Calculate similarity of code and nl directly using tensors
            similarity_tensor = torch.cosine_similarity(code_vec, nl_vec, dim=1)
            similarity.extend(similarity_tensor.cpu().tolist())  # Convert tensor to list for further use
        # 为了方便测试设的break
        # break
    logger.info("Obtain vectors and calculate similarity succeed!")
        
    return similarity
def get_dividing_point(args, model, tokenizer, file_name):
    # get similarity for dataset of filename
    similarity = get_similarity(args, model, tokenizer, file_name)
       # Sort the similarity scores and obtain indices
    sort_ids = np.argsort(similarity)[::-1]

    # Apply Gaussian Mixture Model to find the best dividing point
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    similarity = np.array(similarity)
    gmm.fit(similarity.reshape(-1, 1))  # Reshape for scikit-learn's expectations

    # Predict cluster labels for the data points
    labels = gmm.predict(similarity.reshape(-1, 1))

    # Determine the best dividing point based on the cluster means or other criteria
    # Here we assume the higher similarity group corresponds to 'good' samples and pick the mean of the lower similarity group as dividing point.
    means = gmm.means_.flatten()
    means_sorted = np.sort(means)
    best_dividing_point = means_sorted[0]  # Assuming the first mean corresponds to the 'worse' quality cluster

    logger.info("GMM clustering and obtaining best dividing point succeed!")
    # 输出最佳分割点
    logger.info("Best dividing point: %.4f", best_dividing_point)
    logger.info(f"retain samples length: {len(labels[labels == 0])}")
    
    return similarity,labels



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_file", default=None, type=str,
                        help="file to be calculated and divided.")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")
    parser.add_argument("--model_name_or_path", default=None, type=str, nargs='+',
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # print arguments
    args = parser.parse_args()

    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)
    with open(args.data_file,'r') as f:
            data = json.load(f)
    df = pd.DataFrame(data)
    for model_name in args.model_name_or_path:
        logger.info("loading model %s",model_name)
        # load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = Model(model)
        logger.info("Parameters %s", args)
            
        model.to(args.device)
        # 寻找 dividing point
        similarity,labels = get_dividing_point(args,model,tokenizer,args.data_file)
        # 保存结果到csv文件
        modelname = model_name.replace('\\','/').split('/')[-1]
        df['similarity'+'_'+model_name] = similarity
        df['labels'+'_'+model_name] = labels
        output_file_name = args.data_file.replace('.json','.csv')
    df.to_csv(output_file_name,index=False)       
if __name__ == "__main__":
    main()