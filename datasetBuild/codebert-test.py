from transformers import AutoTokenizer, AutoModel
import torch

# 初始化分词器和模型
tokenizer = AutoTokenizer.from_pretrained("codebert-base")
model = AutoModel.from_pretrained("codebert-base")

# 对自然语言句子进行分词
nl_tokens1 = tokenizer.tokenize("return maximum")
nl_tokens2 = tokenizer.tokenize("return maximum")
# 对代码片段进行分词
code_tokens = tokenizer.tokenize("def max(a,b): if a>b: return a else return b")

# 合并tokens，包括分类符(tokenizer.cls_token)、分隔符(tokenizer.sep_token)和结束符(tokenizer.eos_token)
tokens = [tokenizer.cls_token] + nl_tokens1 + [tokenizer.sep_token]+[tokenizer.cls_token] + nl_tokens2 + [tokenizer.sep_token]

# 将tokens转换为对应的token IDs
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

# 使用模型获取tokens的嵌入表示
context_embeddings = model(torch.tensor(tokens_ids)[None,:])[1]
print(context_embeddings.shape)
print(context_embeddings)