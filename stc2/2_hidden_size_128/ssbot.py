from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch import optim
import torch.nn.functional as F
from IPython import embed

SOS_token = 0
EOS_token = 1
PAD_token = 2

max_length = 42
hidden_size = 128
batch_size = 256
train_num = batch_size * 400
teacher_forcing_ratio = 0.5
#dropout_p = 0.5
learning_rate = 0.05
epochs = 1000
print('max_length: ' + str(max_length))
print('hidden_size: ' + str(hidden_size))
print('batch_size: ' + str(batch_size))
print('learning_rate: ' + str(learning_rate))
print('epochs: ' + str(epochs))

GPUID = 2
USE_CUDA = torch.cuda.is_available()
print('GPUID:' + str(GPUID))
print('USE_CUDA: ' + str(USE_CUDA))

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3   # Count SOS, EOS, and PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    #s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    lines = open('data/sw-100000.txt', encoding='utf-8').read().strip().split('\n')
    #lines = list(set(lines))
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    lines = open('data/ww-100000.txt', encoding='utf-8').read().strip().split('\n')
    #lines = list(set(lines))
    input_w = []
    for l in lines:
        s = l.split('\t')
        input_w.append(normalizeString(s[0]))
    return input_lang, output_lang, pairs, input_w

def prepareData(lang1, lang2, max_length, reverse=False):
    input_lang, output_lang, pairs, input_w = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs, input_w

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        #self.embedding.weight.data.copy_(torch.eye(hidden_size))
        self.embedding.weight.requires_grad = True

        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, -1, self.hidden_size)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if USE_CUDA:
            return result.cuda(GPUID)
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        #self.embedding.weight.data.copy_(torch.eye(hidden_size))
        self.embedding.weight.requires_grad = True
        
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, -1, self.hidden_size)
        for i in range(self.n_layers):
            #output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if USE_CUDA:
            return result.cuda(GPUID)
        else:
            return result

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    for i in range(max_length - 1 - len(sentence.split(' '))):
        indexes.append(PAD_token)
    result = torch.LongTensor(indexes)
    if USE_CUDA:
        return result.cuda(GPUID)
    else:
        return result

def variablesFromPairs(input_lang, output_lang, pairs):
    res = []
    for pair in pairs:
        input_variable = variableFromSentence(input_lang, pair[0])
        target_variable = variableFromSentence(output_lang, pair[1])
        res.append((input_variable, target_variable))
    return res

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=max_length):
    encoder_hidden = encoder.initHidden(batch_size)

    input_variable = Variable(input_variable.transpose(0, 1))
    target_variable = Variable(target_variable.transpose(0, 1))

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)

    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_input = decoder_input.cuda(GPUID) if USE_CUDA else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            topv, topi = decoder_output.data.topk(1)
            decoder_input = Variable(torch.cat(topi))
            decoder_input = decoder_input.cuda(GPUID) if USE_CUDA else decoder_input

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def trainIters(encoder, decoder, epochs, batch_size, print_every, test_every, learning_rate=learning_rate):
    encoder_optimizer = optim.SGD(filter(lambda x: x.requires_grad, encoder.parameters()), lr=learning_rate)
    decoder_optimizer = optim.SGD(filter(lambda x: x.requires_grad, decoder.parameters()), lr=learning_rate)
    criterion = nn.NLLLoss()
    for epoch in range(1, epochs + 1):
        loss = 0
        for batch_x, batch_y in loader:
            loss += train(batch_x, batch_y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        if epoch % print_every == 0:
            print(str(epoch) + ': ' + str(loss))
        if epoch % test_every == 0:
            test(encoder, decoder, epoch)

def evaluate(encoder, decoder, input_variable, max_length=max_length):
    encoder_hidden = encoder.initHidden(1)
    input_variable = Variable(input_variable)
    input_length = input_variable.size()[0]
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda(GPUID) if USE_CUDA else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('EOS')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda(GPUID) if USE_CUDA else decoder_input
        
    return decoded_words

def test(encoder, decoder, epoch, num=100):
    f = open('data/res' + str(epoch) + '.txt', 'w')
    f.write('train_num: ' + str(num) + ' / ' + str(train_num) + '\n\n')
    for i, pair in enumerate(input_pairs[:train_num]):
        if i > num:
            break
        output_words = evaluate(encoder, decoder, variableFromSentence(input_lang, pair[0]).view(-1, 1))
        output_sentence = ' '.join(output_words)
        output_sentence = output_sentence.replace('EOS', '').replace('PAD', '').strip()
        f.write('> ' + input_w[i] + '\n')
        f.write('= ' + pair[1] + '\n')
        f.write('< ' + output_sentence + '\n\n')
    '''
    f.write('test_num: ' + str(num) + ' / ' + str(len(input_pairs) - train_num) + '\n\n')
    for i, pair in enumerate(input_pairs[train_num:]):
        if i > num:
            break
        output_words = evaluate(encoder, decoder, variableFromSentence(input_lang, pair[0]).view(-1, 1))
        output_sentence = ' '.join(output_words)
        output_sentence = output_sentence.replace('EOS', '').replace('PAD', '').strip()
        f.write('> ' + input_w[i] + '\n')
        f.write('= ' + pair[1] + '\n')
        f.write('< ' + output_sentence + '\n\n')
    '''

input_lang, output_lang, input_pairs, input_w = prepareData('s', 'w', max_length)
pairs = variablesFromPairs(input_lang, output_lang, input_pairs)
loader = torch.utils.data.DataLoader(pairs[:train_num], batch_size=batch_size, shuffle=True)

encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = DecoderRNN(hidden_size, output_lang.n_words)

if USE_CUDA:
    encoder = encoder.cuda(GPUID)
    decoder = decoder.cuda(GPUID)

trainIters(encoder, decoder, epochs, batch_size, print_every=epochs / 100, test_every=epochs / 10)
#torch.save(encoder, 'models/encoder.pkl')
#torch.save(decoder, 'models/decoder.pkl')