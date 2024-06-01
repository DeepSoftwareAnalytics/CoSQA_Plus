import concurrent.futures.thread
import time
import requests
import json
import logging
import pandas as pd
import random
import openai
from openai import OpenAI
from datetime import datetime
import sys
import re
import os
from askLLM import askgpt,ask_ollama,ask_text_embedding
import pickle
def judge_match():
    """
    judge_match() 负责调用 askgpt 函数，并获取其返回值，然后将返回值放入表格对象中
    这个相当于主函数
    为了提高速度，这里采用了多线程并发处理
    """
    logging.info("start the judge!")
    input_file = "CoSQA-plus/dataset/augment_codebase.json"
    # output_file = "dataset_annotation_claude3_code_GPT4.csv"
    temp_file = "temp_output.pkl"
    pickle_file = "code_embedding.pkl"
    # 读取之前保存的 csv 文件
    try:
        # df = pd.read_csv(output_file, index_col=0)
        with open(pickle_file, "rb") as file:
            embedding_data = pickle.load(file)
            existing_indexes = set(embedding_data.keys())
    except FileNotFoundError:
        embedding_data = dict()
        existing_indexes = set()

    with open(input_file, "r") as file:
        json_data = json.load(file)
        l = len(json_data)
    max_concurrent_tasks = 500 # 最大并发任务数
    now_index = 0
    # 最大线程数设为50
    with concurrent.futures.thread.ThreadPoolExecutor(max_workers=500) as executor:
        futures = []
        while now_index < l:
            for i in range(max_concurrent_tasks-len(futures)):
                if now_index in existing_indexes:
                    logging.info("Skipping the %s th query as it already exists in the DataFrame", now_index)
                    now_index += 1
                    continue
                if now_index >= l:
                    break
                logging.info("judging the %s th query-code pair", now_index)
                data = json_data[now_index]
                future = executor.submit(judge_task, data, now_index)
                futures.append(future)
                now_index += 1
                if now_index >= l:
                    break
                time.sleep(0.1)
            for future in concurrent.futures.as_completed(futures):
                embedding_current, now_index1 = future.result()
                if embedding_current is None:
                    logging.info("the %s th query/code embedding", now_index1)
                    # 去除失败的futures，防止其阴魂不散
                    futures.remove(future)
                    continue

                embedding_data[now_index1] = embedding_current
                # 保存当前 DataFrame 到表格
                if now_index1%251==0:
                    pickle.dump(embedding_data, open(temp_file, "wb"))
                    os.replace(temp_file, pickle_file)
                    logging.info("save the %s th query/code embedding", now_index1)
                logging.info("finish the %s th query/code embedding", now_index1)
                # 去除已经完成的future，如果不去会重复遍历存在的让表格无限加下去
                futures.remove(future)

        pickle.dump(embedding_data, open(pickle_file, "wb"))
        # os.replace(temp_file, pickle_file)


def judge_task(data, index):
    embedding, model = ask_text_embedding(data["code"])
    # answer_json, model = ask_ollama(prompt)
    if embedding is None:
        return None,index
    return embedding, index


if __name__ == '__main__':
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # 保存日志到文件
    # logging.getLogger().addHandler(logging.FileHandler("log/cosqa_all_llama3_70b_instruct_self_contained_5_judge.log"))
    judge_match()
    mydata={
        'text':'通知',
        'desp':f'{datetime.now()}代码跑完了'
        }
    requests.post('https://wx.xtuis.cn/BgR6QtEhVB1NpWIDRhtvUAl82.send', data=mydata)

    mydata={
        'text':'警告',
        'desp':f'time:{datetime.now()}代码意外中断,并重启运行。错误信息:{sys.exc_info()}'
        }
    requests.post('https://wx.xtuis.cn/BgR6QtEhVB1NpWIDRhtvUAl82.send', data=mydata)
        