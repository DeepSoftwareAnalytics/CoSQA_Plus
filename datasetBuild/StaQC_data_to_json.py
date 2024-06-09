import pickle
import json
import os
from guesslang import Guess



def check_python(code):
    guess = Guess()
    language = guess.language_name(code)
    return language == 'Python'

def write_to_json(idx,code):
    with open("StaQC-code.json", "a+") as f:
        json.dump({"idx": "StaQC-"+str(idx), "code": code}, f)
        f.write(os.linesep)

def check_pickle_file(file_name):
    idx = 0
    with open(file_name, "rb") as f:
        data = pickle.load(f)
        for code in enumerate(data):
            if check_python(code):
                idx += 1
                write_to_json(idx,code)
    print(f"Total number of Python codes: {idx}")

if __name__ == "__main__":
    check_pickle_file("python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle")