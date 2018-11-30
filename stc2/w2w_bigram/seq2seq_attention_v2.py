# -*- coding: utf-8 -*-
# https://arxiv.org/pdf/1409.0473.pdf
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import os
import time
import math
import cPickle as pkl
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import numpy as np
from gensim.models import KeyedVectors
import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from GlobalAttention import GlobalAttention
from IPython import embed

class Lang(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK", 3:"PAD"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        if self.name == 'zh':
            for word in [w for w in sentence]:
                self.addWord(word)
            return
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
    return s
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # return s




def readLangs(lang1, lang2, reverse=False, filename=''):
    print("Reading lines...")

    # Read the file and split into lines
    if filename == '':
        filename = '%s-%s.txt' % (lang1, lang2)
    lines = open(os.path.join(DATA_PATH, filename), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    #pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = [[s for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    filted_pairs = [pair for pair in pairs if filterPair(pair)]
    return filted_pairs


def prepareData(lang1, lang2, reverse=False, filename=''):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse, filename)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)

    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embed_weight=None, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        if embed_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embed_weight))
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)


    def forward(self, input_, hidden=None, cell=None):
        if hidden is None:
            hidden = self.initHidden(input_.size()[0])
            cell = self.initHidden(input_.size()[0])
        embedded = self.embedding(input_)
        output = embedded
        for _ in range(self.n_layers):
            output, (hidden, cell) = self.lstm(output, (hidden, cell))
        return output, hidden, cell

    def initHidden(self, n_sample):
        result = Variable(torch.zeros(1, n_sample, self.hidden_size))
        if USE_CUDA:
            return result.cuda(GPUID)
        else:
            return result

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers, dropout_p, attn_method):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = GlobalAttention(self.hidden_size, method=attn_method)
        self.attn_input_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()
        self.LogSoftmax = nn.LogSoftmax()

    def forward(self, input_, hidden, cell, encoder_hiddens):
        embedded = self.embedding(input_)[:, 0, :]
        embedded = self.dropout(embedded)

        attn_applied, attn_weights = self.attn(hidden.squeeze(0), encoder_hiddens)

        output = torch.cat((embedded, attn_applied), 1)
        output = self.attn_input_combine(output).unsqueeze(1)

        for _ in range(self.n_layers):
            output = self.relu(output)
            output, (hidden, cell) = self.lstm(output, (hidden, cell))

        output = self.LogSoftmax(self.out(output[:, -1, :]))
        return output, hidden, cell, attn_weights

    def initHidden(self, n_sample):
        result = Variable(torch.zeros(1, n_sample, self.hidden_size))
        if USE_CUDA:
            return result.cuda(GPUID)
        else:
            return result


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def indexesFromSentence(lang, sentence,):
    sentence = sentence.split(' ')
    return [lang.word2index.get(word, UNK_TOKEN) for word in sentence]

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_TOKEN)
    return Variable(torch.LongTensor(indexes).view(1, -1))
    

def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

def batchify(pairs):
    print('Crating batchs')
    batchs = {}
    precessed_batch = []

    for pair in pairs:
        n_input, n_target = len(pair[0].split(' ')), len(pair[1].split(' '))
        length = 0
        for b in BUCKETS:
            length = b
            if n_input < b:
                break
        if (length, n_target) not in batchs:
            batchs[(length, n_target)] = []
        pad_input = ' '.join(['PAD'] * (length-n_input)) + ' ' + pair[0]
        batchs[(length, n_target)].append((pad_input,pair[1]))
    print('Length variable batch:',batchs.keys())
    for batch in batchs.values():
        lang1 = [variableFromSentence(input_lang, pair[0]) for pair in batch]
        lang2 = [variableFromSentence(output_lang, pair[1]) for pair in batch]
        for p in zip(torch.cat(lang1, 0).split(BATCH_LENGTH), torch.cat(lang2, 0).split(BATCH_LENGTH)):
            
            precessed_batch.append(p)
            
    return precessed_batch

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=0.5):
    loss = 0

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    n_sample = input_variable.size()[0]
    input_length = input_variable.size()[1]
    target_length = target_variable.size()[1]

    # encoder forward
    encoder_hiddens, encoder_hidden, encoder_cell = encoder(input_variable)

    # prepare decoder input data which the fist input is the index of start of sentence
    decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]*n_sample])).view(-1, 1)
    decoder_input = decoder_input.cuda(GPUID) if USE_CUDA else decoder_input
    decoder_hidden = encoder_hiddens[:, -1, :].unsqueeze(0)
    #decoder_cell = encoder_cell # this is using last hidden state for decoder inital hidden state
    decoder_cell = decoder.initHidden(n_sample) # get cell with zero value

    use_teacher_forcing = True #if random.random() < teacher_forcing_ratio else False


    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(
            decoder_input, decoder_hidden, decoder_cell, encoder_hiddens)

        loss += criterion(decoder_output, target_variable[:, di])

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            decoder_input = target_variable[:, di:di+1]
        else:
            # Without teacher forcing: use its own predictions as the next input
            topv, topi = decoder_output.data.topk(1)
            ni = topi
            decoder_input = Variable(ni).cuda(GPUID) if USE_CUDA else Variable(ni)
            # the decoder is used batching training so can't use EOS_TOKEN to stop decode
            #if ni == EOS_TOKEN:
            #break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def trainEpochs(encoder, decoder, pairs, n_epochs, print_every, plot_every, save_every, learning_rate):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adagrad(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adagrad(decoder.parameters(), lr=learning_rate*0.1)
    
    num_pairs = len(pairs)
    batchs = batchify(pairs[:num_pairs-TESTING_NUM])
    n_batchs = len(batchs)
    criterion = nn.NLLLoss()
    print('total number of batch: %d'%(n_batchs))
    ts = "%d"%(time.time())
    PN = 'BS-'+str(BATCH_LENGTH)+'_HS-'+str(HIDDEN_SIZE)+'_AM-'+str(ATTN_METHOD)\
        +'_DR-'+str(DECODER_DROPOUT)+'_EP-'+str(NUM_EPOCH)+'_LR-'+str(LEARNING_RATE)
    for epoch in range(1, n_epochs + 1):
        # set model for training
        encoder.train()
        decoder.train()
        print('epoch:%d/%d'%(epoch, n_epochs))
        b = 1
        for batch in batchs:
            print('[%d/%d]'%(b,n_batchs),end='\r')
            b += 1
            input_variable = batch[0].cuda(GPUID) if USE_CUDA else batch[0]
            target_variable = batch[1].cuda(GPUID) if USE_CUDA else batch[1]
            loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / (print_every*n_batchs)
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / (plot_every*n_batchs)
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if epoch % save_every ==0:
            # set model for evaluate
            encoder.eval()
            decoder.eval()
            torch.save(encoder, open(os.path.join(SAVE_PATH, ts+'-'+SAVE_PREFIX+'-encoder-'+PN+'epoch'+str(epoch)+'.model'), 'wb'))
            torch.save(attn_decoder, open(os.path.join(SAVE_PATH, ts+'-'+SAVE_PREFIX+'-attn_decoder-'+PN+'epoch'+str(epoch)+'.model'), 'wb'))
            print('All of models are trained and saved to file with prefix %s'%(os.path.join(SAVE_PATH, ts+'-'+SAVE_PREFIX)))
            print('-------testing with training data----------')
            evaluateSpecifiedCases(encoder, attn_decoder, input_lang, output_lang, pairs[0:5], MAX_LENGTH)
            evaluateRandomly(encoder, attn_decoder, input_lang, output_lang, pairs[:num_pairs-TESTING_NUM], MAX_LENGTH, 5)
            print('')
            print('-------testing with testing data-----------')
            evaluateRandomly(encoder, attn_decoder, input_lang, output_lang, pairs[num_pairs-TESTING_NUM:], MAX_LENGTH, 5)

    return plot_losses



def showPlot(points):
    #plt.figure()
    #fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    #loc = ticker.MultipleLocator(base=0.2)
    #ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

def evaluate(encoder, decoder, input_lang, output_lang, max_length, sentence):
    input_variable = variableFromSentence(input_lang, sentence)

    n_sample = input_variable.size()[0]
    input_length = input_variable.size()[1]

    # encoder forward
    input_variable = input_variable.cuda(GPUID) if USE_CUDA else input_variable
    encoder_hiddens, encoder_hidden, encoder_cell = encoder(input_variable)

    # prepare decoder input data which the fist input is the index of start of sentence
    decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]]))  # SOS
    decoder_input = decoder_input.cuda(GPUID) if USE_CUDA else decoder_input
    decoder_hidden = encoder_hiddens[:, -1, :].unsqueeze(0)
    #decoder_cell = encoder_cell
    decoder_cell = decoder.initHidden(n_sample) # get cell with zero value

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, input_length)

    assert max_length > 0
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(
            decoder_input, decoder_hidden, decoder_cell, encoder_hiddens)

        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_TOKEN:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda(GPUID) if USE_CUDA else decoder_input

    return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, max_length, num):
    for _ in range(num):
        pair = random.choice(pairs)
        output_words, attentions = evaluate(encoder, decoder, input_lang, output_lang, max_length, pair[0])
        output_sentence = ' '.join(output_words)
        # try:
        #     print('>', pair[0])
        #     print('=', pair[1])
        #     print('<', output_sentence)
        # except:
        print('>', pair[0].encode('utf-8'))
        print('=', pair[1].encode('utf-8'))
        print('<', output_sentence.encode('utf-8'))
        print('')

def evaluateSpecifiedCases(encoder, decoder, input_lang, output_lang, pairs, max_length):
    for pair in pairs:
        print('>', pair[0].encode('utf-8'))
        output_words, attentions = evaluate(encoder, decoder, input_lang, output_lang, max_length, pair[0])
        output_sentence = ' '.join(output_words)
        print('=', pair[1].encode('utf-8'))
        print('<', output_sentence.encode('utf-8'))
        print('')


def evaluateInteractively(encoder, decoder, input_lang, output_lang, max_length):
    input_ = raw_input('input one sentence..\n')
    input_ = input_.decode('utf-8')

    while True:
        output_words, attentions = evaluate(encoder, decoder, input_lang, output_lang, max_length, input_)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

        input_ = raw_input('input one sentence..\n')
        input_ = input_.decode('utf-8')

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    attention_weight = attentions.numpy()[:, 0:len(input_sentence.split(' '))+1]
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attention_weight, cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
    except:
        print('Does not show poly')

def evaluateAndShowAttention(encoder, decoder, input_lang, output_lang, max_length, input_sentence):
    output_words, attentions = evaluate(
        encoder, decoder, input_lang, output_lang, max_length, input_sentence)
    print('%d input words = %s'%(len(input_sentence.split(' ')), input_sentence))
    print('%d system output words = %s'%(len(output_words[:-1]), ' '.join(output_words[:-1])))
    print('Attention weights:')
    print(attentions)
    showAttention(input_sentence, output_words, attentions)
    return attentions

USE_CUDA = torch.cuda.is_available()

GPUID = 0
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 3
UNK_TOKEN = 2
MAX_LENGTH = 30

BATCH_LENGTH = 500
HIDDEN_SIZE = 64
ATTN_METHOD = 'general' # "dot", "general" or "concat"
DECODER_DROPOUT = 0.1
NUM_LAYER = 1
NUM_EPOCH = 20000
PRINT_EVERY = 1
PLOT_EVERY = 1
SAVE_EVERY = 20
LEARNING_RATE = 0.05
TESTING_NUM = 5
DATA_PATH = 'data'
SAVE_PATH = 'models'
SAVE_PREFIX = 'w2w-bigram-100000'
BUCKETS = [30]
FILE_PATH = 'ww-100000.txt'
EMBEDDING_PATH = ''
EMBEDDING_DIM = 64

def get_args():
    parser = argparse.ArgumentParser(description='seq2seq model with attention')
    parser.add_argument('-c', '--config-path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    if args.config_path:
        with open(args.config_path) as config_file:
            for line in config_file:
                exec(line.strip())

    # NUM_EPOCH = int(raw_input('Epoch: '))
    # data preprocessing
    input_lang, output_lang, pairs = prepareData('word', 'word', False, FILE_PATH)
    num_pairs = len(pairs)

    # build encoder and decoder models
    if EMBEDDING_PATH:
         embedding = KeyedVectors.load(EMBEDDING_PATH)
         embedding_weight = np.zeros([input_lang.n_words , EMBEDDING_DIM])
        
         for index, word in input_lang.index2word.items():
             if word in embedding:
                 embedding_weight[index,] = embedding[word]

         encoder_hidden_size = EMBEDDING_DIM
    
    else:
         embedding_weight=None
         encoder_hidden_size = HIDDEN_SIZE

    encoder = EncoderRNN(input_size=input_lang.n_words,
                         hidden_size= encoder_hidden_size,
                         embed_weight = embedding_weight,
                         n_layers=NUM_LAYER)
    attn_decoder = AttnDecoderRNN(hidden_size=HIDDEN_SIZE,
                                  output_size=output_lang.n_words,
                                  n_layers=NUM_LAYER,
                                  dropout_p=DECODER_DROPOUT,
                                  attn_method=ATTN_METHOD)
    print(encoder)
    print(attn_decoder)
    # set model using GPU's cuda
    if USE_CUDA:
        print('using CUDA models')
        encoder = encoder.cuda(GPUID)
        attn_decoder = attn_decoder.cuda(GPUID)

    

    print('after __coder.train()')
    ts = "%d"%(time.time())
    PN = 'BS-'+str(BATCH_LENGTH)+'_HS-'+str(HIDDEN_SIZE)+'_AM-'+str(ATTN_METHOD)\
                        +'_DR-'+str(DECODER_DROPOUT)+'_EP-'+str(NUM_EPOCH)+'_LR-'+str(LEARNING_RATE)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    pkl.dump((input_lang, output_lang, pairs), open(os.path.join(SAVE_PATH, ts+'-'+SAVE_PREFIX+'.data.pkl'), 'wb'))

    # main training process
    exit()
    plot_losses = trainEpochs(encoder=encoder,
                              decoder=attn_decoder,
                              pairs=pairs,
                              n_epochs=NUM_EPOCH,
                              print_every=PRINT_EVERY,
                              plot_every=PLOT_EVERY,
                              save_every=SAVE_EVERY,
                              learning_rate=LEARNING_RATE)
    
    
    
    #showPlot(plot_losses)

    # set model for evaluation
    encoder.eval()
    attn_decoder.eval()

    print('-------testing with training data----------')
    evaluateRandomly(encoder, attn_decoder, input_lang, output_lang, pairs[:num_pairs-TESTING_NUM], MAX_LENGTH, 1)
    print('')
    print('-------testing with testing data-----------')
    evaluateRandomly(encoder, attn_decoder, input_lang, output_lang, pairs[num_pairs-TESTING_NUM:], MAX_LENGTH, 1)

    #attentions = evaluateAndShowAttention(encoder,attn_decoder,input_lang,output_lang,MAX_LENGTH, random.choice(pairs)[0])
    #plt.matshow(attentions.numpy())

    # save models and data to file
    ts = "%d"%(time.time())
    PN = 'BS-'+str(BATCH_LENGTH)+'_HS-'+str(HIDDEN_SIZE)+'_AM-'+str(ATTN_METHOD)\
                        +'_DR-'+str(DECODER_DROPOUT)+'_EP-'+str(NUM_EPOCH)+'_LR-'+str(LEARNING_RATE)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    pkl.dump((input_lang, output_lang, pairs), open(os.path.join(SAVE_PATH, ts+'-'+SAVE_PREFIX+'.data.pkl'), 'wb'))
    torch.save(encoder, open(os.path.join(SAVE_PATH, ts+'-'+SAVE_PREFIX+'-encoder-'+PN+'.model'), 'wb'))
    torch.save(attn_decoder, open(os.path.join(SAVE_PATH, ts+'-'+SAVE_PREFIX+'-attn_decoder-'+PN+'.model'), 'wb'))
    print('All of models are trained and saved to file with prefix %s'%(os.path.join(SAVE_PATH, ts+'-'+SAVE_PREFIX)))
    print('Interactive!! type a evaluation command as follows:')
    print('attentions = evaluateAndShowAttention(encoder,attn_decoder,input_lang,output_lang,MAX_LENGTH, random.choice(pairs)[0])')
