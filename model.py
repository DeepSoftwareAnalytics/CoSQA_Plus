# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch
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
class GraphCodeBertModel(nn.Module):   
    def __init__(self, encoder):
        super(GraphCodeBertModel, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None): 
        if code_inputs is not None:
            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)
            inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
            return self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[1]
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]
