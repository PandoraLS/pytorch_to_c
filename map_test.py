# -*- coding: utf-8 -*-
# @Time    : 2020/11/23 15:03
# @Author  : sen

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



if __name__ == '__main__':
    vocab = Vocab(min_num=32, max_num=288)
    print()