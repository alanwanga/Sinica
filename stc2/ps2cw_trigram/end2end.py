# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import torch
import os
from seq2seq_attention_v4 import *
import cPickle as pkl
import random
# print 'before'
# res = subprocess.check_output(['python', '-u', 'seq2seq_attention_sense.py'])
# res = subprocess.check_output(['python', '-u', 'seq2seq_attention.py'])
# 
# 
# print res
# print 'after'

from IPython import embed

def end2end(encoder,decoder, input_lang, output_lang, posts, cmnts=[]):
    predicts=[]
    for i,post in enumerate(posts):

        output_words, attentions = evaluate(encoder, decoder, input_lang, output_lang, MAX_LENGTH, post)
        output_sentence = ' '.join(output_words)
        predicts.append(output_sentence)

        print('>', post.encode('utf-8'))
        if cmnts:
            print('=', cmnts[i].encode('utf-8'))
        print('<', output_sentence.encode('utf-8'))
        print('')
    return predicts

def readFromFile(model_file,test_file):
    with open(model_file,'r',encoding='utf-8') as config:
        encoder_filename = config.readline().strip()
        attn_decoder_filename = config.readline().strip()
        data_filename = config.readline().strip()
        gpuid = config.readline().strip()
        cuda_gpu = u'cuda:'+gpuid
        print('----------') 
        #encoder = torch.load(encoder_filename)
        encoder = torch.load(encoder_filename, map_location={'cuda:0':'cpu','cuda:1':'cpu','cuda:2':'cpu','cuda:3':'cpu'})
        encoder = encoder.cuda(GPUID)

        #attn_decoder = torch.load(attn_decoder_filename)
        attn_decoder = torch.load(attn_decoder_filename, map_location={'cuda:0':'cpu','cuda:1':'cpu','cuda:2':'cpu','cuda:3':'cpu'})
        attn_decoder = attn_decoder.cuda(GPUID)
        print(attn_decoder)
        print(encoder)
        data_file = open(data_filename,'rb')
        input_lang, output_lang  = pkl.load(data_file)[0:2]

    posts = []
    cmnts = []
    with open(test_file,'r',encoding='utf-8') as test:
        for line in test.readlines():
            line = line.strip()
            if '\t' in line:
                pair = line.split('\t')
                posts.append(pair[0])
                cmnts.append(pair[1])
            else:
                posts.append(line)
    
    return {"encoder": encoder, "decoder": attn_decoder, "input_lang": input_lang, "output_lang": output_lang,
             "posts": posts, "cmnts": cmnts}

if __name__ == '__main__':
    import sys
    # example: pythone end2end.py model_config.txt sense_word_test.txt
    model_file = sys.argv[1]
    test_file = sys.argv[2]

    material = readFromFile(model_file, test_file)
    results = end2end(material["encoder"], material["decoder"], material["input_lang"], material["output_lang"],
            material["posts"], material["cmnts"])    
    
    if len(sys.argv)>=4:
        output_file = sys.argv[3]
        output = open(output_file, 'w', encoding='utf-8')
        output.write('\n'.join(results))
        output.close()

