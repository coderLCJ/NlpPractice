# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         main
# Description:  
# Author:       Laity
# Date:         2022/3/22
# ---------------------------------------------
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {0: "<SOS>", 1: "<EOS>", 2: "<unk>"}
        self.idx = 3  # Count SOS and EOS

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return 2
        return self.word2idx[word]

    def __len__(self):
        return self.idx


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, seq_input, hidden):
        embedded = self.embedding(seq_input).view(1, seq_input.size(0), self.hidden_size)
        embedded = F.relu(embedded)
        output, hidden = self.gru(embedded, hidden)
        return hidden

    def sample(self, seq_list):
        '''
        v1使用循环方式步进，v2使用gru的自动步进省去循环
        '''
        word_inds = torch.LongTensor(seq_list).unsqueeze(1).to(device)
        h = self.initHidden()
        features = self(word_inds, h)
        return features

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_vocab = output_size
        self.maxlen = 15

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, dec_output, hidden):
        '''
        v2版本为能够训练时能让gru自动步进，省去for循环，均采用【teacher learning】训练
        且forward仅仅返回loss
        '''

        # dec_output: <SOS> xx xx <EOS>
        # dec_input: <SOS> xx xx
        # dec_target: xx xx <EOS>
        dec_input = dec_output[:-1]
        dec_target = dec_output[1:]
        # 【teacher learning】
        # 每一时刻知道当前正确的输入单词 而不是直接拿上一时刻预测的单词做输入 因为可能错的越来越离谱
        dec_input = self.embedding(dec_input).view(1, dec_input.size(0), self.hidden_size)
        dec_input = F.relu(dec_input)

        output, hidden = self.gru(dec_input, hidden)
        output = self.out(output.squeeze(0))

        loss = F.cross_entropy(output.view(-1, self.n_vocab), dec_target.view(-1), reduction='mean')
        # cross_entropy会自动使用logsoftmax，且v2版本修复了v1遇到错误生成EOS而导致的训练bug

        return loss

    def sample(self, pre_hidden):
        word_inputs = torch.tensor(SOS_token, device=device)
        hidden = pre_hidden
        res = [SOS_token]
        for i in range(self.maxlen):
            emb_input = self.embedding(word_inputs).view(1, 1, self.hidden_size)
            emb_input = F.relu(emb_input)
            output, hidden = self.gru(emb_input, hidden)
            output = self.softmax(self.out(output.squeeze(0)))
            topv, topi = output.topk(1, dim=1)
            if topi.item() == EOS_token:
                res.append(EOS_token)
                break
            else:
                res.append(topi.item())
            word_inputs = topi.squeeze().detach()
        return res


lan1 = Vocabulary()
lan2 = Vocabulary()

data = [['你 很 聪明 。', 'you are very wise .'],
        ['我们 一起 打 游戏 。', 'let us play game together .'],
        ['你 太 刻薄 了 。', 'you are so mean .'],
        ['你 完全 正确 。', 'you are perfectly right .'],
        ['我 坚决 反对 妥协 。', 'i am strongly opposed to a compromise .'],
        ['他们 正在 看 电影 。', 'they are watching a movie .'],
        ['他 正在 看着 你 。', 'he is looking at you .'],
        ['我 怀疑 他 是否 会 来', 'i am doubtful whether he will come .']]

for i, j in data:
    lan1.add_sentence(i)
    lan2.add_sentence(j)


def sentence2tensor(lang, sentence):
    indexes = list()
    indexes.append(SOS_token)
    indexes += [lang(word) for word in sentence.split()]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def pair2tensor(pair):
    input_tensor = sentence2tensor(lan1, pair[0])
    target_tensor = sentence2tensor(lan2, pair[1])
    return (input_tensor, target_tensor)


learning_rate = 0.001
hidden_size = 256

encoder = EncoderRNN(len(lan1), hidden_size).to(device)
decoder = DecoderRNN(hidden_size, len(lan2)).to(device)
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=learning_rate)

loss = 0
turns = 200

print_every = 20
training_pairs = [pair2tensor(random.choice(data)) for pair in range(turns)]

for turn in range(turns):
    optimizer.zero_grad()
    loss = 0

    x, y = training_pairs[turn]
    input_length = x.size(0)
    target_length = y.size(0)

    h = encoder.initHidden()
    h = encoder(x, h)

    loss = decoder(y, h)

    print_loss_total = loss.item() / target_length
    if (turn + 1) % print_every == 0:
        print("loss:{loss:,.4f}".format(loss=print_loss_total / print_every))
        print_loss_total = 0

    loss.backward()
    optimizer.step()


def translate(s):
    t = [lan1(i) for i in s.split()]
    t.append(EOS_token)
    f = encoder.sample(t)
    s = decoder.sample(f)
    r = [lan2.idx2word[i] for i in s]
    return ' '.join(r)


for pr in data:
    print('>>', pr[0])
    print('==', pr[1])
    print('result:', translate(pr[0]))
    print()