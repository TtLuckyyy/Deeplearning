import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__  #   obtain the class name
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        print("inital  linear weight ")

class word_embedding(nn.Module):
    def __init__(self,vocab_length , embedding_dim):
        super(word_embedding, self).__init__()
        # 生成 形状为 (vocab_length, embedding_dim) 的随机词向量矩阵，范围在 [-1,1]。
        # 假设 vocab_length=5000, embedding_dim=100，那么 w_embeding_random_intial 就是 (5000, 100) 的 NumPy 数组。
        w_embeding_random_intial = np.random.uniform(-1,1,size=(vocab_length ,embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length,embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))
    def forward(self,input_sentence):
        sen_embed = self.word_embedding(input_sentence)#把input_sentence变成embedding吧
        return sen_embed


class RNN_model(nn.Module):
    def __init__(self, batch_sz ,vocab_len ,word_embedding,embedding_dim, lstm_hidden_dim):
        super(RNN_model,self).__init__()

        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim

        self.rnn_lstm = nn.LSTM(input_size=self.word_embedding_dim,hidden_size=self.lstm_dim,num_layers=2,batch_first=True)
        self.h0= torch.zeros(2, 1, self.lstm_dim).to(device)
        self.c0= torch.zeros(2, 1, self.lstm_dim).to(device)

        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)
        self.apply(weights_init) 
        self.softmax = nn.LogSoftmax(dim=1)  # Specify dim=1 explicitly here
        self.tanh = nn.Tanh()

    def forward(self,sentence,is_test = False):
        # embeding 层
        batch_input = self.word_embedding_lookup(sentence).view(1,-1,self.word_embedding_dim)
        ''''
        lstm：
        输入参数：
            batch_input: 输入的词向量，形状 (batch_size, seq_len, embedding_dim)
            h_0: 初始隐藏状态，形状 (num_layers, batch_size, hidden_size)
            c_0: 初始细胞状态，形状 (num_layers, batch_size, hidden_size)
        输出参数：
            output: 每个时间步的隐藏状态 (batch_size, seq_len, hidden_size)
            {h_n: 最后一个时间步的隐藏状态 (num_layers, batch_size, hidden_size)
            c_n: 最后一个时间步的细胞状态 (num_layers, batch_size, hidden_size)}
        '''
        output, _ = self.rnn_lstm(batch_input, (self.h0, self.c0))
        out = output.contiguous().view(-1,self.lstm_dim)# 防止有permute或者transpose操作，需要contiguous，保证数据在内存连续
        out =  F.relu(self.fc(out))
        out = self.softmax(out)  # 在dim=1维度上应用LogSoftmax
        if is_test:
            prediction = out[ -1, : ].view(1,-1)    
            output = prediction # [vacabu_len] 测试阶段，只需要根据前面字符来预测最后一个时间步的输出作为当前这句话下一个字的预测。 
        else:
           output = out # [seqlen,vocabu_len] 训练阶段，计算整个序列的损失，因此每个时间步的输出都需要去计算损失
        return output