# CoSQA_Plus

## Test on CoSQA +

Our CoSQA + dataset is available in the following link:
https://drive.google.com/drive/folders/10tw69kI0xq5LDU02aJBfk39xpVjdVf7r?usp=drive_link

Place the dataset file in the newly created dataset folder.

test_agent_query json is a query collection

test_agent_codebase json is a code collection

est_agent_pairs json for query-code pairs

test_agent_pairs_true are pairs labeled 1

We provide a test program: evaluation/evaluate_transformer.py

Run command (using CodeBERT as an example):

```Python
python evaluate/run_evalute_transformer.py \
    --model_name_or_path model/codebert-base  \
    --do_zero_shot \
    --do_test \
    --query_pre_process dataset/test_agent_query.json \
    --query_file dataset/test_agent_query.json \
    --codebase_file_pre_process dataset/test_agent_codebase.json \
    --codebase_file dataset/test_agent_codebase_processed.json \
    --true_pairs_file dataset/test_agent_pairs_true.json \
    --eval_batch_size 512 \
    --seed 123456
```

## Use CoSQA + to  fine-tuning

In the dataset link mentioned above, the finetune folder contains the fine-tuned dataset.

Training training dataset

Dev validation set

Test test set

Our test program can also perform model fine-tuning: evaluation/evaluate_transformer.py

Fine-tune command (using CodeBERT as an example):

```Python
python evaluate/run_evalute_transformer.py \
    --do_train \
    --output_dir saved_models/codebert \
    --model_name_or_path model/codebert-base  \
    --query_pre_process dataset/finetune/dev_query.json \
    --query_file dataset/finetune/dev_query_processed.json \
    --codebase_file_pre_process dataset/test_agent_codebase.json  \
    --codebase_file dataset/test_agent_codebase_processed.json \
    --true_pairs_file dataset/test_agent_pairs_true.json \
    --train_data_file dataset/finetune/train_query_code_pairs.json \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 512 \
    --learning_rate 2e-5 \
    --seed 123456
```
