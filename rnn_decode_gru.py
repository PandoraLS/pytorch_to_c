# -*- coding: utf-8 -*-
# @Time    : 2020/7/15 10:44
# @Author  : sen

"""
seq2seq plc:
从C:\Education\code\pytorch_learn2\rnn_decode_single_20200724_1中复制过来
其中 decoder_last.pth也是从C:\Education\code\pytorch_learn2\rnn_decode_single_20200724_1中复制过来的
"""

import os
import re
import random
import unicodedata
import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# import logger

print("start time:  ", time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime(time.time())))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)

class Vocab:
    def __init__(self, min_num, max_num):
        self.token2id = {}
        self.id2token = {}
        self.sos_id = self.add_token(-1)  # 起始符
        self.eos_id = self.add_token(-2)  # 结束符
        self.unk_id = self.add_token(-3)  # unknown符号

        for token in range(min_num, max_num + 1):
            self.add_token(token)

    def add_token(self, token):
        if token in self.token2id:
            _id = self.token2id[token]
        else:
            _id = len(self.token2id)
            self.token2id[token] = _id
            self.id2token[_id] = token
        return _id

    def get_id(self, token):
        return self.token2id.get(token, self.unk_id)

    def get_token(self, id):
        return self.id2token.get(id, '[UNKONWN]')

    def __len__(self):
        return len(self.token2id)


vocab = Vocab(min_num=32, max_num=288)
SRC_length = 8
TAR_length = 4


"""
Model
"""

class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size

        # embedding层的结构，1. 有多少个词，2. 每个词多少维
        self.embedding = nn.Embedding(output_size, hidden_size)
        # GRU的参数: 1. 输入x的维度, 2. 隐藏层状态的维度; 这里都用了hidden_size
        # emb_dim == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        # [batch_size, hidden_size] -> [batch_size, output_size]
        # 这里output_size就是目标语言字典的大小V
        self.out = nn.Linear(hidden_size, output_size)
        # softmax层, 求每一个单词的概率
        self.softmax = nn.LogSoftmax(dim=1)  

    def forward(self, input, hidden):
        # input: [1], 一个单词的下标
        # hidden: [1, 1, hidden_size]
        # embedding(input): [emb_dim]
        output = self.embedding(input).view(1, 1, -1)  # 展开
        # output: [1, 1, emb_dim]
        output = F.relu(output)

        # output: [1, 1, emb_dim]

        # 关于gru的输入输出参数
        # [seq_len, batch_size, input_size],  [num_layers * num_directions, batch_size, hidden_size]
        # output: [1, 1, emb_dim], hidden: [1, 1, hidden_size]
        output, hidden = self.gru(output, hidden)
        # output: [1, 1, hidden_size] # [seq_len, batch, num_directions * hidden_size] # 这里hidden_size == emb_dim
        # output[0]: [1, emb_dim]
        # self.out(output[0]): [1, V]
        # output: [1, V] 值为每个单词的概率
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


"""
训练
"""

def read_dataset(fp, left_count=8, right_count=4):
    pairs = []
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            elems = line.strip().split('\t\t')
            src = [int(_) for _ in elems[:left_count]]
            dst = [int(_) for _ in elems[left_count + 1: left_count + 1 + right_count]]
            pairs.append((src, dst))
    print('load dataset', fp, 'total sample', len(pairs))
    return pairs


def input_to_tensor(inputs):
    ids = [vocab.get_id(_) for _ in inputs]
    return torch.tensor(ids + [vocab.eos_id], dtype=torch.long, device=device).view(-1, 1)


def output_to_token(outputs):
    tokens = [vocab.get_token(_) for _ in outputs[:-1]]
    return tokens


def tensor_from_pair(pair):
    src12, dst4 = pair
    input_tensor = input_to_tensor(src12)
    target_tensor = input_to_tensor(dst4)
    return input_tensor, target_tensor

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, decoder,
          decoder_optimizer, criterion):
    target_length = target_tensor.size(0)
    decoder_optimizer.zero_grad()
    sequence_tensor = torch.cat((input_tensor[:-1,:], target_tensor), 0)  # 去除input_tensor的最后一个eos,然后与target_tensor拼接起来
    sequence_length = sequence_tensor.size(0) # 12 + 1 = 13
    loss = 0
    decoder_input = torch.tensor([[vocab.sos_id]], device=device)
    decoder_hidden = decoder.init_hidden()

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(sequence_length):
            if di < 8:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                decoder_input = sequence_tensor[di]
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                # decoder_output: [1, V] 值为每个单词的概率
                loss += criterion(decoder_output, sequence_tensor[di])
                decoder_input = sequence_tensor[di]
    else:
        # without teacher forcing: use its own predictions as the next input
        for di in range(sequence_length):
            if di < 8:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                decoder_input = sequence_tensor[di]
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, sequence_tensor[di])
                if decoder_input.item() == vocab.eos_id:
                    break

    loss.backward()

    decoder_optimizer.step()

    return loss.item() / target_length


"""
开始训练
"""


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (-%s)' % (as_minutes(s), as_minutes(rs))


# pairs = read_dataset(r'/Users/seenli/Documents/workspace/code/pytorch_learn/seq2seq/hidden_20200717_single/data/pitch_2to1_clean.txt',
#                      left_count=8, right_count=4)
# pairs_val = read_dataset(r'/Users/seenli/Documents/workspace/code/pytorch_learn/seq2seq/hidden_20200717_single/data/pitch_2to1_clean_head2000.txt',
#                          left_count=8, right_count=4)
# pairs = read_dataset(r'/Users/seenli/Documents/workspace/code/pytorch_learn/seq2seq/hidden_20200717_context/data/pitch_2to1_clean_swap.txt',
#                      left_count=8, right_count=4)
# pairs_val = read_dataset(r'/Users/seenli/Documents/workspace/code/pytorch_learn/seq2seq/hidden_20200717_context/data/pitch_2to1_clean_swap_head2000.txt',
#                          left_count=8, right_count=4)
pairs = None
pairs_val = None

min_loss_avg = float('inf')  # 用于判别最小值的loss


def evaluate(decoder, src_ids, src_length=SRC_length + 1, tar_length=TAR_length + 1):
    sequence_length = src_length + tar_length - 1 # 长度为13
    with torch.no_grad():
        input_tensor = input_to_tensor(src_ids)
        decoder_input = torch.tensor([[vocab.sos_id]], device=device)
        decoder_hidden = decoder.init_hidden()
        decoded_words = []

        for di in range(sequence_length):
            if di < 8: #
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                decoder_input = input_tensor[di]
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)

                if topi.item() == vocab.eos_id:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(topi.item())
                decoder_input = topi.squeeze().detach()  # detach from history as input
        decoded_words = output_to_token(decoded_words)
        return decoded_words


def evaluate_randomly(decoder, n=3000, has_attention=False):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        if has_attention:
            raise NotImplemented
        else:
            output = evaluate(decoder, pair[0])
        print('<', output)
        print('')


def evaluate_head(decoder, n=1000, has_attention=False):
    # eval 测试集中的前n行
    print("evaluate validation dataset ...")
    for i in range(n):
        pair = pairs_val[i]
        print(pair[0], ' > ', pair[1])
        if has_attention:
            raise NotImplemented
        else:
            output = evaluate(decoder, pair[0])
        print('<', output)
        print('')


def train_iters(decoder, n_iters, print_every=1000,
                plot_every=100, eval_every=100,
                learning_rate=0.001, has_attention=False):
    print("learning_rate:", learning_rate)
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # reset each print_every
    plot_loss_total = 0  # reset each plot_every

    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [random.choice(pairs) for i in range(n_iters)]

    # nn.NLLLoss(): The negative log likelihood loss. It is useful to train a classification problem with C classes.
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = tensor_from_pair(training_pairs[iter - 1])
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        if has_attention:
            raise NotImplemented
        else:
            loss = train(input_tensor, target_tensor, decoder,
                         decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))
            global min_loss_avg
            if print_loss_avg < min_loss_avg:
                min_loss_avg = print_loss_avg
                torch.save(decoder.state_dict(), 'decoder_best.pth')
                print('save best model, iter: %d, min_loss: %.4f' % (iter, min_loss_avg))

            # 每1000iter就会将最新的存储一下
            torch.save(decoder.state_dict(), 'decoder_last.pth')
            print_loss_total = 0

        if iter % plot_every == 0:
            plot_loss_avg = print_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if iter % eval_every == 0:
            print('evaluate random 3 sentence with last model ...')
            decoder.load_state_dict(torch.load('decoder_last.pth'))
            evaluate_randomly(decoder, n=3)


hidden_size = 128
print("hidden_size = ", hidden_size)


print("len(vocab) = ", len(vocab))
decoder = DecoderGRU(hidden_size=hidden_size, output_size=len(vocab)).to(device)
print(decoder)
# 输出网络参数量
from torchsummaryX import summary
inputs = torch.ones((1, 1), dtype=torch.long)
hiddens = torch.ones(1, 1, hidden_size)
summary(decoder, inputs, hiddens)
# train_iters(decoder=decoder,
#             n_iters=1600000,
#             print_every=1000,
#             eval_every=10000) # 每eval_every小测一下

print("end time:  ", time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime(time.time())))

