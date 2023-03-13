import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class My_Model(nn.Module):
    
    """定义Transformer网络结构，只使用了encoder
    Args:   words_num：用到glove中的单词数
            d_model：词嵌入维度
            weight：词嵌入的向量对应表
            n_layers：encoder的个数
            n_heads：多头注意力的个数
            d_k：key向量的维数
            d_v：value向量的维数
            d_inner：前馈神经网络的中间维度
            d_out：输出的维度， 即情感的类别个数
            n_position：句子的最大单词数

    """
    def __init__(self, words_num, d_model, weight, n_layers, n_heads, d_k, d_v, d_inner, d_out, dropout=0.1, n_position=200):
        
        super(My_Model, self).__init__()
        self.Encoder = Encoder(words_num, d_model, weight, n_layers, n_heads, d_k, d_v, d_inner, dropout, n_position)
        self.fc = nn.Linear(d_model, d_out, bias=True)
        


    def forward(self, input):
        """input:[batchsize, seq_len]"""
        output = self.Encoder(input) ##output:[batchsize, seq_len, d_model]
        output = self.fc(output[:, 0, :].squeeze(1))
        return output



class position_encode(nn.Module):
    """定义位置编码"""
    def __init__(self, d_model, max_sequence_len, dropout=0.1):
        super(position_encode, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        ##计算pos_encode的参数，i的取值为 0,1,2,3 ... dim/2 - 1 (当dim为奇数时，i可以取到dim/2) 
        position =np.array([[pos / np.power(10000, (2 * i) / d_model ) for i in range(int(d_model / 2) + 1)]
                             for pos in range(max_sequence_len)])
        
        ##初始化pos_encode
        pos_encode = np.zeros((max_sequence_len, d_model))
        ##even index为sin
        pos_encode[:, ::2] = np.sin(position[:, list(range(*pos_encode[0, ::2].shape))]) 
        ##odd index为cos
        pos_encode[:, 1::2] = np.cos(position[:, list(range(*pos_encode[0, 1::2].shape))])

        self.pos_encoding = torch.FloatTensor(pos_encode).cuda()

    def forward(self, input): 
            """input维度[batchsize, sentence_length, embedding_dim]"""
            output = input + self.pos_encoding[:input.shape[1], :] ##利用了广播机制实现维度不同的tensor相加
            return self.dropout(output)


def get_attn_pad_mask(input):
    '''
    mask掉padding词
    input: [batch_size, seq_len]
    由于padding由序号0标记，因此通过与0比较找到padding词的位置
    '''
    batch_size, seq_len = input.size()
    # eq(zero) is PAD token
    pad_attn_mask = input.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len], False is masked
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]


class ScaledDotProductAttention(nn.Module):
    """通过Q，K，V矩阵，计算出各词向量的加权和"""
    def __init__(self, d_model,attn_dropout=0.1):
          super(ScaledDotProductAttention, self).__init__()
          self.d_model = d_model
          self.dropout = nn.Dropout(attn_dropout)


    def forward(self, Q, K, V, attn_mask):
         """Q: [batchsize, nheads, len_q, d_k]  query 和 key的维度是相同的
            K: [batchsize, nheads, len_k, d_k]
            V: [batchsize, nheads, len_k(=len_q), d_v]
            attn_mask: [batch_size, n_heads, seq_len, seq_len]"""
         ##Q,K的转置（后两个维度转置）相乘得到各个value的权重：[batchsize, nheads, len_q, len_k]
         attn = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_model)
         attn.masked_fill_(attn_mask, -1e9) ##用一个充分小的数代替padding
         attn = self.dropout(F.softmax(attn, dim=-1))
         output = torch.matmul(attn, V) ##value加权求和
         return output, attn

class MultiHeadAttn(nn.Module):
    """多头注意力机制，产生W_q, W_k, W_v矩阵来生成词向量的query，key和value"""
    def __init__(self, d_model, key_dim, value_dim, num_heads, dropout=0.1):
          super(MultiHeadAttn, self).__init__()
          self.d_k = key_dim ##key的维度，也等同于query的维度
          self.d_v = value_dim ##value的维度
          self.n_head = num_heads ##多头注意力的头数
          ##全连接层linear(no bias)相当于矩阵乘法
          self.W_q = nn.Linear(d_model, self.d_k * self.n_head, bias=False)
          self.W_k = nn.Linear(d_model, self.d_k * self.n_head, bias=False)
          self.W_v = nn.Linear(d_model, self.d_v * self.n_head, bias=False)
          self.fc = nn.Linear( self.n_head * self.d_v, d_model, bias=False)

          self.attention = ScaledDotProductAttention(d_model, dropout)
          self.layer_norm = nn.LayerNorm(d_model)
          self.dropout = nn.Dropout(dropout)

    def forward(self, input, attn_mask):
        
        residual, batchsize = input, input.shape[0]
        Q = self.W_q(input).view(batchsize, -1, self.n_head, self.d_k).transpose(1,2) ##矩阵Q：[batchsize, nhead, sequence_len, dim_k] 
        K = self.W_k(input).view(batchsize, -1, self.n_head, self.d_k).transpose(1,2) ##矩阵K：[batchsize, nhead, sequence_len, dim_k]
        V = self.W_v(input).view(batchsize, -1, self.n_head, self.d_v).transpose(1,2) ##矩阵V：[batchsize, nhead, sequence_len, dim_v] 
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)              # For head axis broadcasting attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = self.attention(Q, K, V, attn_mask)          # context: [batch_size, n_heads, len_q, d_v]
                                                                                 # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batchsize, -1, self.n_head * self.d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.dropout(self.fc(context))                                                # [batch_size, len_q, d_model]
        return self.layer_norm(output + residual), attn




class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_in, d_out, dropout=0.1 ):
          super(FeedForward, self).__init__()
          
          self.fc = nn.Sequential(
               nn.Linear(d_in, d_out, bias = False),
               nn.ReLU(),
               nn.Linear(d_out, d_in, bias=False)
               
          )
          self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
          self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs):
        residual = inputs
        outputs = self.fc(inputs)
        outputs = self.dropout(outputs)
        return self.layer_norm(residual + outputs) ## [batchsize, sequence_len, d_model]
    

class EncoderBlock(nn.Module):
    """单个encoder模块"""
    def __init__(self, d_model, d_key, d_value, nhead, d_inner, dropout=0.1):
          super(EncoderBlock, self).__init__()
        #   self.d_model = d_model ##词嵌入的维度
        #   self.d_k = d_key ##key向量的维度
        #   self.d_v = d_value ##value向量的维度
        #   self.nhead = nhead ##多头注意力的个数
          self.attn = MultiHeadAttn(d_model,d_key, d_value, nhead, dropout=dropout) ##多头注意力机制网络
          self.feedforward = FeedForward(d_model,d_inner,dropout ) ##前馈神经网络

    def forward(self, inputs, attn_mask):
        outputs, attn = self.attn(inputs, attn_mask)
        outputs = self.feedforward(outputs)
        return outputs, attn
    
class Encoder(nn.Module):
    """整体encoder，包含多个encoder block
    Args:   words_num：用到glove中的单词数
            d_model：词嵌入维度
            weight：词嵌入的向量对应表
            n_layers：encoder的个数
            n_heads：多头注意力的个数
            d_k：key向量的维数
            d_v：value向量的维数
            d_inner：前馈神经网络的中间维度
            n_position：句子的最大单词数
            """
    def __init__(self, words_num, d_model, weight, n_layers, n_heads, d_k, d_v, d_inner, dropout=0.1, n_position=200):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=words_num, embedding_dim=d_model, _weight=weight)
        self.pos_encode = position_encode(d_model, n_position, dropout)
        self.dropout = nn.Dropout(dropout)
        ##encoder序列
        self.encoders_stack = nn.ModuleList([EncoderBlock(d_model, d_k, d_v, n_heads, d_inner, dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, input):
         attn_mask = get_attn_pad_mask(input)
         ##词嵌入
         output = self.embedding(input)
         ##位置编码
         output = self.dropout(self.pos_encode(output))
         output = self.layer_norm(output)
         for encoder in self.encoders_stack:
              output, _ = encoder(output, attn_mask)
         return output
