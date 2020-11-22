# -*- coding: utf-8 -*-
# @Time    : 2020/11/21 20:12
# @Author  : sen

"""
参考链接：https://www.cnblogs.com/carsonzhu/p/11197048.html
利用Tracing将模型转化为 Torch Script
"""

# TODO 下面用到的evaluate需要用c语言来实现

import torch
import decode_rnn
from rnn_decode_gru import evaluate

model_path = 'decoder_last.pth'

# 模型实例化
model = decode_rnn.DecoderGRU(hidden_size=128, output_size=260)
model.load_state_dict(torch.load(model_path), strict=True)
# 若在C++中只进行模型的推断，保存模型之前要model.eval()。否则在C++中无法进行训练模式与测试模式的切换
model.eval()
print(model)

# An example input you would normally provide to your model's forward() method.
# inputs = torch.ones((1, 1), dtype=torch.long)
inputs = torch.ones((1, 1), dtype=torch.long)
hiddens = torch.ones(1, 1, 128)  # hidden_size=128

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, (inputs, hiddens))
output = traced_script_module(inputs, hiddens)
print("output: ", output)
print("type(output): ", type(output))
print("output[0].shape: ", output[0].shape)
print("output[1].shape: ", output[1].shape)

# traced_script_module.save("decoder_last.pt")

# 简单测试一下预测效果, 预测效果基本符合预期，这里seq2seq准是没有学习到
src_list = [132, 132, 132, 132, 129, 128, 127, 126]       # predict:[125, 125, 125, 125]
# src_list = [61, 60, 60, 60, 63, 65, 65, 67]     # predict: [67, 67, 67, 67]
predict = evaluate(model, src_list)
print("predict: ", predict)

