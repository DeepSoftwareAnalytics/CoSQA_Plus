# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
from datetime import datetime
import sys 
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import UniXcoderModel,CodeBertModel,JinaV3Model,AllMiniLMModel
from sentence_transformers import SentenceTransformer
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)
from transformers import AutoTokenizer, AutoModel
logger = logging.getLogger(__name__)
from tqdm import tqdm
from metrics import CalculateMNDCG,CalculateMRR
os.environ['TOKENIZERS_PARALLELISM'] = 'true'  # 或者 'false'
###############################################################################
# 目前没有改的：auto-label
models_need_tokenize = ['unixcoder-base','codebert-base','codet5p-110m-embedding']
# custom 表示在程序中进行提前手动tokenize
# auto 表示在程序中进行自动tokenize
model_type = {
    'unixcoder-base':'custom',
    'codebert-base':'custom',
    'codet5p-110m-embedding':'custom',
    'e5-base-v2': 'auto',
    'jina-embeddings-v3': 'auto',
    'all-MiniLM-L12-v2': 'auto',
    'multilingual-e5-large': 'auto',
    'all-mpnet-base-v2': 'auto',
}
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 pair_idx,
                 query_idx,
                 code_idx,
                 label
    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        ################# add:
        self.pair_idx = pair_idx
        self.query_idx = query_idx
        self.code_idx = code_idx
        self.label = label

def convert_examples_to_features(js,tokenizer,args):
    
    """convert examples to token ids"""
    model_name = args.model_name_or_path.split('/')[-1]
    code = (" ".join(js["code"].split()))
    nl = (" ".join(js["query"].split()))
    if model_type[model_name] == 'auto':
        # if model_name == "all-MiniLM-L12-v2":
        #     code = tokenizer(code, padding=True, truncation=True, return_tensors='pt')
            # nl = tokenizer(nl, padding=True, truncation=True, return_tensors='pt')
        # 不需要进行tokenzize
        return InputFeatures(code,'',nl,'',js['pair-idx'],js['query-idx'],js['code-idx'],js['label'])
    if model_name == "unixcoder-base":
        code_tokens = tokenizer.tokenize(code)[: args.code_length - 4]
        code_tokens = (
            [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token]
            + code_tokens
            + [tokenizer.sep_token]
        )
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = args.code_length - len(code_ids)
        code_ids += [tokenizer.pad_token_id]*padding_length
    elif model_name == "codebert-base":
        code_tokens = tokenizer.tokenize(code)[:args.code_length-2]
        code_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = args.code_length - len(code_ids)
        code_ids += [tokenizer.pad_token_id]*padding_length
    elif model_name == "codet5p-110m-embedding":
        code_tokens = tokenizer.tokenize(code)[: args.code_length]
        code_ids = tokenizer.encode(code)[: args.code_length]
        padding_length = args.code_length - len(code_ids)
        code_ids += [tokenizer.pad_token_id]*padding_length
    # elif model_name == "all-MiniLM-L12-v2":
    #     code_tokens = ''
    #     code_ids = tokenizer(code, padding=True, truncation=True, return_tensors='pt')
    # elif model_name == "stella_en_400M_v5":
    
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
    # elif model_name == "all-MiniLM-L12-v2":
    #     nl_tokens = ''
    #     nl_ids = tokenizer(nl, padding=True, truncation=True, return_tensors='pt')
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['pair-idx'],js['query-idx'],js['code-idx'],js['label'])
    

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        self.model_name = args.model_name_or_path.split('/')[-1]
        with open(file_path) as f:
            data = json.load(f)
        for js in tqdm(data):
            self.examples.append(convert_examples_to_features(js,tokenizer,args))     
        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))                             
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):  
        if model_type[self.model_name] == 'custom':
            return (torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids))
        else:
            return (self.examples[i].code_tokens,self.examples[i].nl_tokens)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer):
    """ Train the model """
    #get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    # InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['pair-idx'],js['query-idx'],js['code-idx'],js['label'])
    model_name = args.model_name_or_path.split('/')[-1]

    model.train()
    tr_num,tr_loss,best_mrr = 0,0,0 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)
            #get code and nl vectors
            # code_vec = model(code_inputs)
            # nl_vec = model(nl_inputs)
            if model_name == 'unixcoder-base' or 'codebert-base':
                code_vec = model(code_inputs)
            elif model_name == 'codet5p-110m-embedding':
                code_vec = model(code_inputs)[0]
                
            if model_name == 'unixcoder-base' or 'codebert-base':
                nl_vec = model(nl_inputs)
            elif model_name == 'codet5p-110m-embedding':
                nl_vec = model(nl_inputs)[0]
            
            #calculate scores and loss
            scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores*20, torch.arange(code_inputs.size(0), device=scores.device))
            
            #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate    
        results = evaluate(args, model, tokenizer,args.query_file, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        # if results['eval_mrr']>best_mrr:
        if results['eval_ndcg']>best_mrr:
            best_mrr = results['eval_ndcg']
            logger.info("  "+"*"*20)  
            logger.info("  Best NDCG:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

         
def evaluate(args, model, tokenizer,file_name,eval_when_training=False):
    query_dataset = TextDataset(tokenizer, args, file_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    code_dataset = TextDataset(tokenizer, args, args.codebase_file)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    

    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    code_vecs = None
    nl_vecs = None
    model_name = args.model_name_or_path.split('/')[-1]
    # if os.path.exists(f'{model_name}_mmrr_code_vecs.pkl'):
    #     with open(f'{model_name}_mmrr_code_vecs.pkl', 'rb') as f:
    #         code_vecs = pickle.load(f)
    # if os.path.exists(f'{model_name}_mmrr_nl_vecs.pkl'):
    #     with open(f'{model_name}_mmrr_nl_vecs.pkl', 'rb') as f:
    #         nl_vecs = pickle.load(f)
    if nl_vecs is None:
        nl_vecs = []
        for batch in tqdm(query_dataloader):
            if model_type[model_name] == 'custom':
                nl_inputs = batch[1].to(args.device)
            else:
                nl_inputs = list(batch[1])
            with torch.no_grad():
                if model_name in ['unixcoder-base','codebert-base','codet5p-110m-embedding','all-MiniLM-L12-v2','multilingual-e5-large',"all-mpnet-base-v2"]:
                    nl_vec = model(nl_inputs)
                elif model_name == 'jina-embeddings-v3':
                    nl_vec = model(nl_inputs,task_type='query')
                # elif model_name == 'all-MiniLM-L12-v2':
                #     nl_vec = model.encode(nl_inputs)
                nl_vecs.append(nl_vec.cpu().numpy()) 
        nl_vecs = np.concatenate(nl_vecs,0)
    if code_vecs is None:
        code_vecs = []
        for batch in tqdm(code_dataloader):
            if model_type[model_name] == 'custom':
                code_inputs = batch[0].to(args.device)    
            else:
                code_inputs = list(batch[0])
            with torch.no_grad():
                if model_name in ['unixcoder-base','codebert-base','codet5p-110m-embedding','all-MiniLM-L12-v2','multilingual-e5-large','all-mpnet-base-v2']:
                    code_vec = model(code_inputs)
                elif model_name == 'jina-embeddings-v3':
                    code_vec = model(code_inputs,task_type='code')
                # elif model_name == 'all-MiniLM-L12-v2':
                #     code_vec = model.encode(code_inputs)
                code_vecs.append(code_vec.cpu().numpy())
        code_vecs = np.concatenate(code_vecs,0)
    model.train()
        
    
    # pickle.dump(code_vecs,open(f'{model_name}_mmrr_code_vecs.pkl','wb'))
    # pickle.dump(nl_vecs,open(f'{model_name}_mmrr_nl_vecs.pkl','wb'))
    logger.info("embedding done and saved!")
    scores = np.matmul(nl_vecs,code_vecs.T)
    logger.info("scores done!")
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    logger.info("sort done!")
    # sort_ids不是idx，只是code投射下来的排名
    sort_idxs = []
    for sort_id in tqdm(sort_ids):
        sort_idx = []
        for i in sort_id[:1000]:
            sort_idx.append(code_dataset.examples[i].code_idx)
        sort_idxs.append(sort_idx)

    
    
    # 要获取sort_ids对应的所有query-idxs
    query_idxs = []
    for example in tqdm(query_dataset.examples):
        query_idxs.append(example.query_idx)
        
    # 计算mmrr    
    # 不应该用传进来的file_name,这里要用所有正确的pair来评测（因为不止一条对的，传进来的file_name可能只有一条对的）
    logger.info("calculating mmrr...")
    # mmrr = CalculateMMRR(sort_idxs,args.true_pairs_file,query_idxs,k=10)
    mrr,map,precision,recall = CalculateMRR(sort_idxs,args.true_pairs_file,query_idxs,k=10)
    # result = {
    #     "eval_mrr":float(mrr),
    #     "eval_map":float(map),
    #     "eval_precision":float(precision),
    #     "eval_recall":float(recall),
    #     "eval_mmrr":float(mmrr)
    # }
    ndcg = CalculateMNDCG(sort_idxs,args.true_pairs_file,query_idxs,k=10)
    result = {
        "eval_mrr":float(mrr),
        "eval_map":float(map),
        "eval_precision":float(precision),
        "eval_recall":float(recall),
        "eval_ndcg":float(ndcg)
    }
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'result/{model_name}_result_{current_time}.json', 'w') as f:
        json.dump(result,f,indent=4)
    return result

# query.json和codebase在使用前需要转换
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
    parser.add_argument("--train_data_file", default=None, type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
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
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")     
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")      

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)
    model_name = args.model_name_or_path.split('/')[-1]
    # load model and tokenizer
    sentence_model = ['all-MiniLM-L12-v2','multilingual-e5-large','all-mpnet-base-v2']
    if model_name in sentence_model:
        model = SentenceTransformer(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model_name_or_path,trust_remote_code=True)
    model_map = {
        'unixcoder-base': UniXcoderModel,
        'codebert-base': CodeBertModel,
        'jina-embeddings-v3': JinaV3Model,
        'all-MiniLM-L12-v2': AllMiniLMModel,
        'multilingual-e5-large': AllMiniLMModel,
        'all-mpnet-base-v2': AllMiniLMModel,
    }
    if(model_map.get(model_name) is not None):
        model = model_map[model_name](model)
    # model = Model(model)
    logger.info("Training/evaluation parameters %s", args)
    
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Process codebase
    pre_process(args)
    

    # Training
    if args.do_train:
        train(args, model, tokenizer)
      
    # Evaluation
    results = {}
    if args.do_eval:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model.module if hasattr(model, 'module') else model  
            model_to_load.load_state_dict(torch.load(output_dir))    
        model.to(args.device)
        result = evaluate(args, model, tokenizer,args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))
            
    if args.do_test:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model.module if hasattr(model, 'module') else model 
            model_to_load.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer,args.query_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))


if __name__ == "__main__":
    main()



