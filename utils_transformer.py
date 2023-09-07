# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:29:48 2023

@author: SESA608167
"""
import torch
import math
import numpy as np
from torch.nn import functional

class MultiheadAttention(torch.nn.Module):
    def __init__(self, d_k, d_model, n_heads):
        super().__init__()
        
        #Assume dv= dk
        self.d_v= d_k
        self.d_k= d_k
        self.n_heads= n_heads
        
        #Initialize weights:
        self.keys= torch.nn.Linear(d_model, d_k*n_heads)
        self.value= torch.nn.Linear(d_model, d_k*n_heads)
        self.query= torch.nn.Linear(d_model, d_k*n_heads)
        
        #Final linear layer
        self.fin_layer= torch.nn.Linear(d_k*n_heads, d_model)
        
    def forward(self, q, k, v, mask= None):
        q= self.query(q)  # N , tx , d_model (h*d_k)
        k= self.keys(k)  # N , tx , d_model (h*d_k)
        v= self.value(v)  # N , tx , d_model (h*d_v)
        
        N= q.shape[0]
        T1= q.shape[1] 
        
        #CHange the shape
        # (N, T1, heads * d_k) -> (N, T1, heads , d_k)
        # (N, T1, heads , d_k) -> (N, heads, T1 , d_k)
        q= q.view(N, T1, self.n_heads, self.d_k).transpose(1,2)
        k= k.view(N , T1, self.n_heads, self.d_k).transpose(1,2)
        v= v.view(N, T1, self.n_heads, self.d_k).transpose(1,2)
        
        #Compute our equation
        # 1) QK^t (N, heads, T1, d_k) * (N, heades, d_k, T1) --> (N, heades, T1, T1) 
        
        attn_weights= q @ k.transpose(-2,-1) / np.sqrt(self.d_k)
        
        # mask is a tensor of 1's and 0's of size n,t
        if mask is not None:
            attn_weights= attn_weights.masked_fill(
                mask[:, None, None, :]== 0, float('-inf')   )
        
        attn_weightsfin= functional.softmax(attn_weights, dim= -1)
        # (N, heads, T1, T1) * (N, heads, T1 , d_k) -> (N, heads, T1, d_k)
        A2= attn_weightsfin @ v
        #reshape it back
        A2= A2.transpose(1, 2) # (N, heads, T1, d_k) --> (N, T1, heads, d_k)
        A2= A2.contiguous().view(N, T1, self.d_k*self.n_heads)  # (N, T1, heads, d_k) -> (N, T1, heads * d_k)
        
        #projection
        return self.fin_layer(A2)

class TransformadorBlock(torch.nn.Module):
    def __init__(self, d_k, d_model, n_heads, dropout_prob=0.1):
        super().__init__()
   # this module will normalize over the last dimension which is expected to be of that specific size.     
        self.layern1= torch.nn.LayerNorm(d_model) 
        self.layern2= torch.nn.LayerNorm(d_model)
        self.mha_created=  MultiheadAttention( d_k, d_model, n_heads)
        self.ann= torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model*4),
            torch.nn.GELU(),
            torch.nn.Linear(d_model*4, d_model),
            torch.nn.Dropout(dropout_prob)  # Add regularization
        )
        self.dropoutlay= torch.nn.Dropout(p= dropout_prob)
        
        # X is input sequence of size N, T1, d_model
    def forward(self, x, mask=None):
        # Pass x as key, query and value
        x= self.layern1(x + self.mha_created(x,x,x, mask) )
        # X alone inside the equation is the residual connection
        x= self.layern2(x + self.ann(x))
        x= self.dropoutlay(x)
        return x
    
    
class PositionalEncodingsyeah(torch.nn.Module):
    def __init__(self, d_model, max_len=2048, dropout_prob=0.1):
        super().__init__()
        self.dropout= torch.nn.Dropout(p= dropout_prob)
        
        # List of integers from 0 to max len
        position= torch.arange(max_len).unsqueeze(1)
        exp_term= torch.arange(0, d_model, 2)
        div_term= torch.exp(exp_term * (-np.log(10000.0)/d_model) )
        pe= torch.zeros(1, max_len, d_model)
        pe[0,:,0::2]= torch.sin(position*div_term)
        pe[0,:,1::2]= torch.cos(position*div_term)
        self.register_buffer('pe',pe)
    
    def forward(self, x):
        # Shape N, T1, d_model
        x= x+ self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class Encoder_Smash(torch.nn.Module):
    def __init__(self, vocab_size, max_len, d_k,
                d_model, n_heads, n_layers, n_classes, dropout_prob):
        super().__init__()
        
        self.embedding= torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoding= PositionalEncodingsyeah(d_model, max_len, dropout_prob)
        transformer_blocks= [
            TransformadorBlock(d_k, d_model, n_heads, dropout_prob
                              ) for _ in range (n_layers)
        ]
        self.all_transformer_blocks= torch.nn.Sequential(*transformer_blocks)
        self.layer_norm= torch.nn.LayerNorm(d_model)
        self.final_classif= torch.nn.Linear(d_model, n_classes)
        
    def forward(self, x, mask=None):
        
        x= self.embedding(x)
        x= self.pos_encoding(x)
        for block in self.all_transformer_blocks:
            x= block(x, mask)
        # many to one (x has shape N, T1, D)
        x= x[:,0,:] # make it (N, 1 ,D)
        x= self.layer_norm(x)
        x= self.final_classif(x)
        
        return x

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len):
        super().__init__()
        
        #Assume dv= dk
        self.d_v= d_k
        self.d_k= d_k
        self.n_heads= n_heads
        
        #Initialize weights:
        self.keys= torch.nn.Linear(d_model, d_k*n_heads)
        self.value= torch.nn.Linear(d_model, d_k*n_heads)
        self.query= torch.nn.Linear(d_model, d_k*n_heads)
        
        #Final linear layer
        self.fin_layer= torch.nn.Linear(d_k*n_heads, d_model)
        
        #causal mask
        cm= torch.tril(torch.ones(max_len, max_len))
        self.register_buffer(
        'causal_mask', cm.view(1,1,max_len, max_len)
        )
        
    def forward(self, q, k, v, pad_mask= None):
        q= self.query(q)  # N , tx , d_model (h*d_k)
        k= self.keys(k)  # N , tx , d_model (h*d_k)
        v= self.value(v)  # N , tx , d_model (h*d_v)
        
        N= q.shape[0]
        T1= q.shape[1] 
        
        #CHange the shape
        # (N, T1, heads * d_k) -> (N, T1, heads , d_k)
        # (N, T1, heads , d_k) -> (N, heads, T1 , d_k)
        q= q.view(N, T1, self.n_heads, self.d_k).transpose(1,2)
        k= k.view(N , T1, self.n_heads, self.d_k).transpose(1,2)
        v= v.view(N, T1, self.n_heads, self.d_k).transpose(1,2)
        
        #Compute our equation
        # 1) QK^t (N, heads, T1, d_k) * (N, heades, d_k, T1) --> (N, heades, T1, T1) 
        
        attn_weights= q @ k.transpose(-2,-1) / math.sqrt(self.d_k)
        
        # mask is a tensor of 1's and 0's of size n,t
        if pad_mask is not None:
            attn_weights= attn_weights.masked_fill(
                pad_mask[:, None, None, :]== 0, float('-inf')   )
        attn_weights= attn_weights.masked_fill(
        self.causal_mask[:,:,:T1, :T1]==0, float('-inf'))
        
        attn_weightsfin= functional.softmax(attn_weights, dim= -1)
        # (N, heads, T1, T1) * (N, heads, T1 , d_k) -> (N, heads, T1, d_k)
        A2= attn_weightsfin @ v
        #reshape it back
        A2= A2.transpose(1, 2) # (N, heads, T1, d_k) --> (N, T1, heads, d_k)
        A2= A2.contiguous().view(N, T1, self.d_k*self.n_heads)  # (N, T1, heads, d_k) -> (N, T1, heads * d_k)
        
        #projection
        return self.fin_layer(A2)
    
class TransformadorBlock2(torch.nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len,dropout_prob=0.1):
        super().__init__()
   # this module will normalize over the last dimension which is expected to be of that specific size.     
        self.layern1= torch.nn.LayerNorm(d_model) 
        self.layern2= torch.nn.LayerNorm(d_model)
        self.mha_created=  CausalSelfAttention( d_k, d_model, n_heads, max_len)
        self.ann= torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model*4),
            torch.nn.GELU(),
            torch.nn.Linear(d_model*4, d_model),
            torch.nn.Dropout(dropout_prob)  # Add regularization
        )
        self.dropoutlay= torch.nn.Dropout(p= dropout_prob)
        
        # X is input sequence of size N, T1, d_model
    def forward(self, x, pad_mask=None):
        # Pass x as key, query and value
        x= self.layern1(x + self.mha_created(x,x,x, pad_mask) )
        # X alone inside the equation is the residual connection
        x= self.layern2(x + self.ann(x))
        x= self.dropoutlay(x)
        return x
    
class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, max_len, d_k, d_model, 
                n_heads, n_layers, dropout_prob):
        super().__init__()
        
        self.embedding= torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoding= PositionalEncodingsyeah(d_model, max_len, dropout_prob)
        transformer_blocks= [
        TransformadorBlock2(
        d_k, d_model, n_heads, max_len, dropout_prob
        )
        for _ in range(n_layers)]
        
        self.all_transformer_blocks2= torch.nn.Sequential(*transformer_blocks)
        self.layer_norm= torch.nn.LayerNorm(d_model)
        self.final_linearlay= torch.nn.Linear(d_model, vocab_size)
        
    def forward(self, x, pad_mask=None):
        x= self.embedding(x)
        x= self.pos_encoding(x)
        for block in self.all_transformer_blocks2:
            x= block(x)
        
        x= self.layer_norm(x)
        x= self.final_linearlay(x)  # many to many
        
        return x