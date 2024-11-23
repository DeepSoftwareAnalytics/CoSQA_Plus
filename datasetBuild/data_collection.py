from process_data import (
    CSN_ORIGIN_JSON_FILE,
    CSN_CODE_FILE,
    STAQC_CODE_FILE,
    CSN_STAQC_CODE_FILE,
    COSQA_PAIRS_FILE,
    QUERY_FILE,
)
import json


def process_CSN():
    """
    `process_CSN` reads data from CSN json file,
    and extract the code and url from it.
    """

    with open(CSN_ORIGIN_JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    new_data = []
    for i, item in enumerate(data):
        new_data.append(
            {"idx": "CSN-" + str(i), "code": item["function"], "url": item["url"]}
        )
    with open(CSN_CODE_FILE, "w") as f:
        json.dump(new_data, f)


def merge_CSN_and_StaQC():
    with open(CSN_CODE_FILE, "r") as f:
        CSN_data = json.load(f)
    with open(STAQC_CODE_FILE, "r") as f:
        StaQC_data = json.load(f)
    new_data = []
    index = 0
    for i, item in enumerate(CSN_data):
        new_data.append(
            {
                "idx": index,
                "code": item["code"],
                "source": "CSN",
            }
        )
        index += 1
    for i, item in enumerate(StaQC_data):
        new_data.append(
            {
                "idx": index,
                "code": item["code"],
                "source": "StaQC",
            }
        )
        index += 1
    for i in range(3):
        print(f"处理完毕，第{i+1}个样例：{new_data[i]}")
    with open(CSN_STAQC_CODE_FILE, "w") as f:
        json.dump(new_data, f)


def process_query():
    with open(COSQA_PAIRS_FILE, "r") as f:
        data = json.load(f)
    new_data = []
    for i, item in enumerate(data):
        new_data.append({"query-idx": i, "query": item["doc"]})
    with open(QUERY_FILE, "w") as f:
        json.dump(new_data, f)
