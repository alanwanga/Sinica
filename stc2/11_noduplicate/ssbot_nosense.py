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
import pickle
from IPython import embed
from gensim.models import KeyedVectors

SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3

min_freq = 3 # set UNK if word2count is under threshold
max_freq = 4 # keep pair if total num of UNK is under threshold
hidden_size = 128
max_length = 20
batch_size = 32
train_num = batch_size * 2500
teacher_forcing_ratio = 0.5
#dropout_p = 0.5
learning_rate = 0.01
epochs = 10000

log_file = open('data/log_nosense.txt', 'w')
def log(s):
    print(s)
    log_file.write(str(s) + '\n')

log('hidden_size: ' + str(hidden_size))
log('max_length: ' + str(max_length))
log('batch_size: ' + str(batch_size))
log('train_num: ' + str(train_num))
log('learning_rate: ' + str(learning_rate))
#log('epochs: ' + str(epochs))

GPUID = 1
USE_CUDA = torch.cuda.is_available()
print('GPUID:' + str(GPUID))
print('USE_CUDA: ' + str(USE_CUDA))

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK", 3:"PAD"}
        self.n_words = 4

    def addSentence(self, sentence, unk_list=[]):
        for word in sentence.split(' '):
            if word not in unk_list:
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
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    return unicodeToAscii(s.lower().strip())

def readLangs(lang1, lang2, reverse=False):
    log("Reading lines...")
    lines = open('data/ww-rl.txt', encoding='utf-8').read().strip().split('\n')
    s = set()
    pairs = []
    for pair in [[normalizeString(s) for s in l.split('\t')] for l in lines]:
        if pair[0] not in s:
            s.add(pair[0])
            pairs.append(pair)
    if reverse:
        pairs = [list(reversed(pair)) for pair in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    lines = open('data/ww-rl.txt', encoding='utf-8').read().strip().split('\n')
    s = set()
    input_w = []
    for l in lines:
        w = l.split('\t')
        if w[0] not in s:
            s.add(w[0])
            input_w.append(normalizeString(w[0]))
    return input_lang, output_lang, pairs, input_w

def filterPair(pair, unk_list):
    if len(pair[0].split(' ')) < max_length and len(pair[1].split(' ')) < max_length:
        return len(set(pair[0].split(' ')).intersection(unk_list)) + len(set(pair[1].split(' ')).intersection(unk_list)) < max_freq
    return False

def filterPairs(pairs, unk_list): 
    return [pair for pair in pairs if filterPair(pair, unk_list)]

def prepareData(lang1, lang2, max_length, reverse=False):
    input_lang, output_lang, pairs, input_w = readLangs(lang1, lang2, reverse)
    log("Read %s sentence pairs" % len(pairs))
    freq_count = {}
    for pair in pairs:
        for word in pair[0].split(' '):
            if word in freq_count:
                freq_count[word] += 1
            else:
                freq_count[word] = 1
        for word in pair[1].split(' '):
            if word in freq_count:
                freq_count[word] += 1
            else:
                freq_count[word] = 1
    unk_list = []
    for word in freq_count:
        if freq_count[word] < min_freq:
            unk_list.append(word)
    pairs = filterPairs(pairs, set(unk_list))
    log("Trimmed to %s sentence pairs" % len(pairs))
    try:
        input_lang = pickle.load(open("input_lang_nosense.pkl", "rb"))
        output_lang = pickle.load(open("output_lang_nosense.pkl", "rb"))
    except:
        log("Counting words...")
        for pair in pairs:
            input_lang.addSentence(pair[0], unk_list)
            output_lang.addSentence(pair[1], unk_list)
        pickle.dump(input_lang, open("input_lang_nosense.pkl", "wb"))
        pickle.dump(output_lang, open("output_lang_nosense.pkl", "wb"))
    log("Counted words:")
    log(input_lang.n_words)
    log(output_lang.n_words)
    return input_lang, output_lang, pairs, input_w

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embed_weight=None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        if embed_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embed_weight))
        self.embedding.weight.requires_grad = True

        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_, hidden):
        output = self.embedding(input_).view(1, -1, self.hidden_size)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if USE_CUDA:
            return result.cuda(GPUID)
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding.weight.requires_grad = True
        
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input_, hidden):
        output = self.embedding(input_).view(1, -1, self.hidden_size)
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
    return [lang.word2index.get(word, UNK_token) for word in sentence.split(' ')]

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
    result = []
    for pair in pairs:
        input_variable = variableFromSentence(input_lang, pair[0])
        target_variable = variableFromSentence(output_lang, pair[1])
        result.append((input_variable, target_variable))
    return result

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
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    for epoch in range(1, epochs + 1):
        loss = 0
        for batch_x, batch_y in loader:
            loss += train(batch_x, batch_y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        if epoch % print_every == 0: 
            log(str(epoch) + ': ' + str(loss))
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

def test(encoder, decoder, epoch, num=50):
    f = open('data/epoch_nosense' + str(epoch) + '.txt', 'w')
    for i, pair in enumerate(input_pairs[:train_num]):
        if i > num:
            break
        output_words = evaluate(encoder, decoder, variableFromSentence(input_lang, pair[0]).view(-1, 1))
        output_sentence = ' '.join(output_words)
        output_sentence = output_sentence.replace('EOS', '').replace('PAD', '').strip()
        f.write('> ' + input_w[i] + '\n')
        f.write('= ' + pair[1] + '\n')
        f.write('< ' + output_sentence + '\n\n')
    f.write('============================================================================='+ '\n\n')
    for i, pair in enumerate(input_pairs[train_num:]):
        if i > num:
            break
        output_words = evaluate(encoder, decoder, variableFromSentence(input_lang, pair[0]).view(-1, 1))
        output_sentence = ' '.join(output_words)
        output_sentence = output_sentence.replace('EOS', '').replace('PAD', '').strip()
        f.write('> ' + input_w[i] + '\n')
        f.write('= ' + pair[1] + '\n')
        f.write('< ' + output_sentence + '\n\n')

input_lang, output_lang, input_pairs, input_w = prepareData('s', 'w', max_length)
pairs = variablesFromPairs(input_lang, output_lang, input_pairs)
loader = torch.utils.data.DataLoader(pairs[:train_num], batch_size=batch_size, shuffle=True)

EMBEDDING_PATH = ''
#EMBEDDING_PATH = 'skipgram_w2v_ASBC_300d.vec'
if EMBEDDING_PATH:
     embedding = KeyedVectors.load(EMBEDDING_PATH)
     #embedding_weight = np.zeros([input_lang.n_words, EMBEDDING_DIM])
     embedding_weight = np.zeros([input_lang.n_words, hidden_size])
     for index, word in input_lang.index2word.items():
         if word in embedding:
             embedding_weight[index,] = embedding[word]
     #encoder_hidden_size = EMBEDDING_DIM
else:
     embedding_weight = None
     #encoder_hidden_size = HIDDEN_SIZE

encoder = EncoderRNN(input_lang.n_words, hidden_size, embed_weight = embedding_weight)
decoder = DecoderRNN(hidden_size, output_lang.n_words)

if USE_CUDA:
    encoder = encoder.cuda(GPUID)
    decoder = decoder.cuda(GPUID)

trainIters(encoder, decoder, epochs, batch_size, print_every=1, test_every=10)
#torch.save(encoder, 'encoder.pkl')
#torch.save(decoder, 'decoder.pkl')