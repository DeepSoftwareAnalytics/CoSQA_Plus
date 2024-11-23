import json
import random
import pandas as pd
from process_data import QUERY_FILE,STAQC_CODE_FILE
import pickle
import re
def select_1000_query():
    QUERY_1000_FILE = "CoSQA-plus/dataset/query_1000.json"
    with open(QUERY_FILE, "r") as f:
        query_json = json.load(f)
    # 随机挑选1000条
    select_query = random.sample(query_json, 1000)
    with open(QUERY_1000_FILE, "w") as f:
        json.dump(select_query, f)


def select_n_query(num):
    with open(QUERY_FILE, "r") as f:
        query_json = json.load(f)
    # 随机挑选1000条
    select_query = random.sample(query_json, num)
    with open(f"CoSQA-plus/dataset/query_{num}.json", "w") as f:
        json.dump(select_query, f)


def select_n_pairs(num):
    with open("CoSQA-plus/dataset/stage_2_final/final_query_code_pairs.json", "r") as f:
        pairs = json.load(f)

    select_pairs = random.sample(pairs, num)
    with open(f"CoSQA-plus/dataset/human_query_code_pairs_{num}.json", "w") as f:
        json.dump(select_pairs, f)

def cut_code_embedding_pkl():
    with open("CoSQA-plus/dataset/text_embedding_large_code_embedding.pkl", "rb") as f:
        code_embedding = pickle.load(f)
    with open("CoSQA-plus/dataset/stage_2_final/codebase.json", "r") as f:
        codebase = json.load(f)
    cut_length = len(codebase)
    code_embedding_copy = code_embedding.copy()
    for key in code_embedding_copy.keys():
        if key >= cut_length:
            del code_embedding[key]
    print(f"cut code embedding length: {len(code_embedding)}")
    with open("CoSQA-plus/dataset/code_embedding.pkl", "wb") as f:
        pickle.dump(code_embedding, f)

def remove_code_2000():
    with open(STAQC_CODE_FILE, "r") as f:
        code_json = json.load(f)


def remove_no_answer_row(input_file, processed_file):
    df = pd.read_csv(input_file, index_col=0)
    # 从answer列中origin_answer提取
    # 使用正则表达式匹配 "judgement": "([^"]+)"
    remove_index = []
    for index, row in df.iterrows():
        data_str = row["origin_answer"]
        match = re.search(r'("judgement"|"judgment"): "([^"]+)"', data_str)
        if match:
            pass
            # print(f"Match found for pair-index {index}")
        else:
            remove_index.append(index)
            print(f"No match found for pair-index {index}")
    df.drop(remove_index, inplace=True)
    df.to_csv(processed_file)
    print(f"processed file saved: {processed_file} len:{len(df)}")


def remove_empty_row(input_file, processed_file):
    df = pd.read_csv(input_file, index_col=0)
    df = df.dropna(subset=["origin_answer"])
    df.to_csv(processed_file)
    print(f"processed file saved: {processed_file} len:{len(df)}")


def drop_n_pair_index(input_file, processed_file, pair_index):
    df = pd.read_csv(input_file, index_col=0)
    df = df.drop(df[df["pair-index"] == pair_index].index)
    df.to_csv(processed_file)
    print(f"processed file saved: {processed_file} len:{len(df)}")


def change_to_no(input_file, processed_file, pair_index):
    df = pd.read_csv(input_file, index_col=0)
    df.at[pair_index, "judgement"] = "no"
    df.to_csv(processed_file)
    print(f"processed file saved: {processed_file} len:{len(df)}")

        
def df_to_json(pickle_file, json_file):
    """
    read a DataFrame from a pickle file and saves it as a JSON file.

    :param pickle_file: pickle file that contains the DataFrame you want to convert to JSON.
    :param json_file: the file path where you want to save the DataFrame as a JSON file.
    """
    df = pd.read_pickle(pickle_file)
    df.to_json(json_file, orient="records")


def json_to_csv(json_file, df_file):
    """
    convert a JSON file to a CSV file.

    :param json_file: JSON file to be converted to a CSV file.
    :param df_file: the file path to save the converted DataFrame as a CSV file.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.to_csv(df_file, index=False)

def excel_to_json(excel_path, output_path=None):
    """
    Convert a specified path Excel (.xlsx) file to a JSON file.

    parameters:
        excel_path (str): The path to the Excel file.
        output_path (str, optional): The path to the output JSON file. If not provided, return a JSON string instead of writing to a file.

    returns:
        If `output_path` specifies a file path, write the JSON file and return None;
        otherwise, return a JSON string.
    """
    df = pd.read_excel(excel_path)
    json_data = df.to_json(orient='records')

    if output_path is not None:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_data)
        print(f"Excel file '{excel_path}' converted to JSON and saved at '{output_path}'.")
    else:
        print(f"Excel file '{excel_path}' successfully converted to JSON.")
        return json_data

