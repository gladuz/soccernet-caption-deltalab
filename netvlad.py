
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class QFormerVideoPooling(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, num_layers=4, dropout=0.0, num_tokens = 8):
        super(QFormerVideoPooling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tranformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True), num_layers=num_layers)
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, input_dim))
        self.pos_encoder = PositionalEncoding(input_dim, max_len=60)
        
    
    def forward(self, x):
        # x [BS, T, D]
        BS, T, D = x.size()
        x = self.pos_encoder(x)
        queries = self.query_tokens.expand(BS, -1, -1)
        queries = self.pos_encoder(queries)
        x = self.tranformer_decoder(queries, x) # [BS, 8, D]
        return x


class TransformerVideoPooling(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=6, num_layers=2, dropout=0.1):
        super(TransformerVideoPooling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pos_encoder = PositionalEncoding(30, max_len=1000)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=30, nhead=num_heads, dropout=dropout, batch_first=True), num_layers=num_layers)
        self.transformer_2 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=dropout, batch_first=True), num_layers=num_layers)
        self.qformer = QFormerVideoPooling(input_dim, output_dim, num_heads, num_layers, dropout, num_tokens=8)
    
    def forward(self, x):
        # x [BS, T, D]
        x = x.transpose(1, 2)
        BS, T, D = x.size()
        x = self.pos_encoder(x) # [BS, D, T]
        x = self.transformer(x) # [BS, D, T]
        x = self.pos_encoder(x)
        x = x.transpose(1, 2)
        x = self.transformer_2(x)
        x = self.qformer(x)
        return x   


class NetVLAD(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter((1/math.sqrt(feature_size))
                *th.randn(feature_size, cluster_size))
        self.clusters2 = nn.Parameter((1/math.sqrt(feature_size))
                *th.randn(1, feature_size, cluster_size))

        self.add_batch_norm = add_batch_norm
        self.out_dim = cluster_size*feature_size

    def forward(self,x):
        # x [BS, T, D]
        max_sample = x.size()[1]

        # LOUPE
        if self.add_batch_norm: # normalization along feature dimension
            x = F.normalize(x, p=2, dim=2)

        x = x.reshape(-1,self.feature_size)
        assignment = th.matmul(x,self.clusters) 

        assignment = F.softmax(assignment,dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        a_sum = th.sum(assignment,-2,keepdim=True)
        a = a_sum*self.clusters2

        assignment = assignment.transpose(1,2)

        x = x.view(-1, max_sample, self.feature_size)
        vlad = th.matmul(assignment, x)
        vlad = vlad.transpose(1,2)
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)
        
        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size*self.feature_size)
        vlad = F.normalize(vlad) # (T x D) -> 1 x D_c

        return vlad


class NetRVLAD(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetRVLAD, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter((1/math.sqrt(feature_size))
                *th.randn(feature_size, cluster_size))
        # self.clusters2 = nn.Parameter((1/math.sqrt(feature_size))
        #         *th.randn(1, feature_size, cluster_size))
        # self.clusters = nn.Parameter(torch.rand(1,feature_size, cluster_size))
        # self.clusters2 = nn.Parameter(torch.rand(1,feature_size, cluster_size))

        self.add_batch_norm = add_batch_norm
        # self.batch_norm = nn.BatchNorm1d(cluster_size)
        self.out_dim = cluster_size*feature_size
        #  (+ 128 params?)
    def forward(self,x):
        max_sample = x.size()[1]

        # LOUPE
        if self.add_batch_norm: # normalization along feature dimension
            x = F.normalize(x, p=2, dim=2)

        x = x.reshape(-1,self.feature_size)
        assignment = th.matmul(x,self.clusters)

        assignment = F.softmax(assignment,dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        # a_sum = th.sum(assignment,-2,keepdim=True)
        # a = a_sum*self.clusters2

        assignment = assignment.transpose(1,2)

        x = x.view(-1, max_sample, self.feature_size)
        rvlad = th.matmul(assignment, x)
        rvlad = rvlad.transpose(-1,1)

        # vlad = vlad.transpose(1,2)
        # vlad = vlad - a

        # L2 intra norm
        rvlad = F.normalize(rvlad)
        
        # flattening + L2 norm
        rvlad = rvlad.reshape(-1, self.cluster_size*self.feature_size)
        rvlad = F.normalize(rvlad)

        return rvlad


if __name__ == "__main__":
    vlad = NetVLAD(cluster_size=64, feature_size=512)

    feat_in = torch.rand((3,120,512))
    print("in", feat_in.shape)
    feat_out = vlad(feat_in)
    print("out", feat_out.shape)
    print(512*64)
