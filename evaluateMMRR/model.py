# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch    
class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)

class UniXcoderModel(nn.Module):   
    def __init__(self, encoder):
        super(UniXcoderModel, self).__init__()
        self.encoder = encoder
      
    def forward(self, inputs=None): 
        outputs = self.encoder(inputs,attention_mask=inputs.ne(1))[0]
        outputs = (outputs*inputs.ne(1)[:,:,None]).sum(1)/inputs.ne(1).sum(-1)[:,None]
        return torch.nn.functional.normalize(outputs, p=2, dim=1)
            
class CodeBertModel(nn.Module):
    def __init__(self, encoder):
        super(CodeBertModel, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[1]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)

class TextEmbeddingModel(nn.Module):
    def __init__(self, vector_dim):
        super(TextEmbeddingModel, self).__init__()
        self.linear_layer = nn.Linear(in_features=1024, out_features=vector_dim)
        
        # Load state dict for the linear layer
        vector_linear_directory = f"2_Dense_{vector_dim}"
        vector_linear_dict = {
            k.replace("linear.", ""): v for k, v in
            torch.load(f"{vector_linear_directory}/pytorch_model.bin").items()
        }
        self.linear_layer.load_state_dict(vector_linear_dict)

    def forward(self, inputs, attention_mask=None):
        # if attention_mask is None:
        #     raise ValueError("Attention mask must be provided.")
        
        # Forward pass through the model
        with torch.no_grad():
            # Assuming self.model is defined outside this class and passed as an argument
            # 注意这个模型tokenize会用0来填充而不是1
            attention_mask = inputs.ne(0)
            last_hidden_state = self.model(inputs, attention_mask=attention_mask)[0]
            
            # Mask the padded tokens

            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            # Sum the hidden states over the sequence length and divide by the number of tokens
            query_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            
            # Apply the linear transformation and normalize
            text_vectors = self.linear_layer(text_vectors)
            text_vectors = torch.nn.functional.normalize(text_vectors, p=2, dim=1)
            
        return text_vectors
