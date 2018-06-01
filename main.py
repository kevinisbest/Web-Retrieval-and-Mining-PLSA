# -*- coding: UTF-8 -*-  
import numpy as np
import math
import sys
import json
import csv
import time
import re
import os
import codecs
import jieba
from numpy import zeros, int8, log
from pylab import random
from argparse import ArgumentParser
from operator import itemgetter
import xml.etree.cElementTree as ET
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

StopWord_Path = 'stop.txt'
puncs = [u'，', u'?', u'@', u'!', u'$', u'%', u'『', u'』', u'「', u'」', u'＼', u'｜', u'？', u' ', u'*', u'(', u')', u'~', u'.', u'[', u']', u'\\n','u\n',u'1',u'2',u'3',u'4',u'5',u'6',u'7',u'8',u'9',u'0', u'。',u'"']

def Read_Doc(Doc_Path):

    # read the stopword
    print('Reading StopWord... ')
    with open(str(StopWord_Path),encoding='utf-8') as f:
        stopwords = [line.strip() for line in f]
    f.close()

    # read the doc
    print('Reading Doc... ')
    Doc = []
    with open(str(Doc_Path),encoding="utf-8") as f :
        Docs = [line.strip('\n') for line in f]
        for i, doc in enumerate(Docs):
            if i != 0:
                print('Now is NO. '+str(i))
                Doc.append(doc[len(str(i-1))+1:])
    f.close()

    # number of documents
    N = len(Doc)

    wordCounts_in_doc = []
    total_wordCounts = {}
    # word2id = {}
    Vocab_List = []
    # id2word = {}
    currentId = 0

    for doc in Doc:
        for punc in puncs: 
            doc = doc.replace(punc,' ')
        segList = list(jieba.cut(doc))
        tmp_wordCount = {}
        for word in segList:
            word = word.lower().strip()
            if len(word) >1 and not re.search('[0-9]', word) and word not in stopwords:

                if word not in Vocab_List:
                    Vocab_List.append(word)
                try:
                    tmp_wordCount[word]
                except:
                    tmp_wordCount[word] = 1
                else:
                    tmp_wordCount[word] += 1

                try:
                    total_wordCounts[word]
                except:
                    total_wordCounts[word] = 1
                else:
                    total_wordCounts[word] += 1

        wordCounts_in_doc.append(tmp_wordCount)

    # filter rare word
    for key, value in total_wordCounts.items():
        if value < 200: # 要加 're' 'subject' 'com' 'edu '到stop list
            Vocab_List.remove(key)

    # length of dictionary
    M = len(Vocab_List)

    # generate the document-word matrix
    X = np.zeros([N, M])
    for word in Vocab_List:
        j = Vocab_List.index(word)
        for i in range(0, N):
            if (word in wordCounts_in_doc[i]):
                # print(i,j)
                X[i, j] = wordCounts_in_doc[i][word]

    return N, M, X, Vocab_List

# Initial normalization
def Initial(lamda, theta):

    lambaRowSum = lamda.sum(axis=1)
    new_lamda = lamda / lambaRowSum[:, np.newaxis]

    thetaRowSum = theta.sum(axis=1)
    new_theta = theta / thetaRowSum[:, np.newaxis]

    return new_lamda, new_theta

def E_Step(theta, lamda, p):
    for i in range(N):
        p[i, :, :] = theta.transpose() * lamda[i, :]
        for j in range(M):
            s = p[i, j, :].sum()
            if s == 0:
                p[i, j, :] = np.ones(K)
        row_sum = p[i, :, :].sum(axis=1)
        p[i, :, :] = p[i, :, :] / row_sum[:, np.newaxis]
    return p

def M_Step(theta, lamda, p, X):

    for k in range(K):
        for j in range(M):
            theta[k, j] = np.dot(X[:, j], p[:, j, k])
        s = theta[k, :].sum()
        if s == 0:
            theta[k, :] = np.ones(M)
    row_sum = theta.sum(axis=1)
    theta = theta / row_sum[:, np.newaxis]

    for i in range(N):
        for k in range(K):
            lamda[i, k] = np.dot(X[i, :], p[i, :, k])
        s = lamda[i, :].sum()
        if s == 0:
            lamda[i, :] = np.ones(K)
    row_sum = lamda.sum(axis=1)
    lamda = lamda / row_sum[:, np.newaxis]
    # print("M end")
    return theta, lamda


def LogLikelihood(theta, lamda, X):
    loglikelihood = 0
    for i in range(0, N):
        for j in range(0, M):
            s = np.dot(theta[:, j], lamda[i, :])
            if s > 0:
                loglikelihood += X[i, j] * np.log(s)

    return loglikelihood

def OutPut():
    
    # topic-word distribution
    filename = Save_dir+'/topicWordDistribution.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:  

        for i in range(0, K):
            tmp = ''
            for j in range(0, M):
                tmp += str(theta[i, j]) + ' '
            file.write(tmp + '\n')
    file.close()
    
    # dictionary
    filename = Save_dir+'/dictionary.dic'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        for i in range(0, M):
            file.write(Vocab_List[i] + '\n')
        # file.write(id2word[i] + '\n')
    file.close()
    
    
    # top words of each topic
    degrade = {}
    filename = Save_dir+'/topics.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        for i in range(0, K):
            group_score = np.zeros([len(WordsInGroup)])
            topicword = []
            ids = theta[i, :].argsort()
            for j in ids:
                # topicword.insert(0, id2word[j])
                topicword.insert(0, Vocab_List[j])
            tmp = ''
            tmp_list = topicword[0:min(30, len(topicword))]
            for word in tmp_list:

                tmp += word + ' '

            for groupID, keywords in WordsInGroup.items():
                for topic_word in keywords:
                    # if topic_word in tmp_list:
                    #     group_score[groupID]+=1
                    for word in tmp_list:
                        w1 = wordnet.synsets(topic_word)
                        w2 = wordnet.synsets(word)
                        if len(w2)==0 or len(w1)==0: # if this word not in wordnet , continue next word
                            continue
                        sim_score = w1[0].wup_similarity(w2[0])
                        if sim_score is None: # if this word not match the type
                            continue
                        if sim_score > 0.5:
                            # print(topic_word,word)
                            group_score[groupID]+=1
            group_score = np.argsort(-group_score)             
            tmp+=str(group_score[0])
            degrade[i] = group_score[0]
            file.write(tmp + '\n')
    file.close()

    # document-topic distribution
    filename = Save_dir+'/docTopicDistribution.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write('doc_id,class_id\n')
        for i in range(0, N):
            tmp = lamda[i][:]
            tmp = np.argsort(-tmp)

            file.write(str(i)+','+str(degrade[tmp[0]]) + '\n')
    file.close()

    # save two array
    np.save(Save_dir+'/theta.npy', theta)
    np.save(Save_dir+'/lamda.npy', lamda)

# extend the seed word
def Group(Group_Path,dic_flag):
    WordsInGroup = {}
    with open(Group_Path) as f :
        groups = [line.strip() for line in f]
        for i, group in enumerate(groups):
            if i != 0:
                group = group.split(',')
                synonyms = []
                if dic_flag:
                    group = [group[2]]+group[1].split('.')[1:]
                    for word in group:
                        for syn in wordnet.synsets(word):
                            for lemma in syn.lemmas():
                                currentword = lemma.name()
                                if currentword in Vocab_List:
                                    if currentword not in synonyms:
                                        synonyms.append(currentword)
                else:
                    synonyms = group[2]
                WordsInGroup[i-1] = synonyms
    f.close()
    G = len(WordsInGroup)
    return WordsInGroup, G

# only count word 
def word_count_classifer(output_file):
    doc_group = np.zeros(N)
    for i in range(N):
        gw_cnt = np.zeros(G)
        for j in range(G):
            currentGroupWords = WordsInGroup[j]
            group_word_len = len(currentGroupWords)
            for word in currentGroupWords:
                gw_cnt[j] += X[i, Vocab_List.index(word)]
            gw_cnt[j] = gw_cnt[j] / group_word_len
        doc_group[i] = np.argmax(gw_cnt)
    
    filename = Save_dir+'/'+output_file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f_output = open(filename, 'w')
    f_output.write("doc_id,class_id\n")

    for i in range(N):
        f_output.write(str(i) + "," + str(int(doc_group[i])) + "\n")

    f_output.close()

# use plsa matrix
def plsa_classifer(output_file):
    prob_doc_word = np.dot(lamda, theta)
    doc_group = np.zeros(N)
    for i in range(N):
        gw_prob = np.zeros(G)
        for j in range(G):
            currentGroupWords = WordsInGroup[j]
            group_word_len = len(currentGroupWords)
            if group_word_len!=0:
                for word in currentGroupWords:
                    gw_prob[j] += prob_doc_word[i, Vocab_List.index(word)]
                gw_prob[j]/= group_word_len
            else:
                gw_prob[j] = 0
        doc_group[i] = np.argmax(gw_prob)

    filename = Save_dir+'/'+output_file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f_output = open(filename, 'w')
    f_output.write("doc_id,class_id\n")

    for i in range(N):
        f_output.write(str(i) + "," + str(int(doc_group[i])) + "\n")

    f_output.close()

def main():
    parser = ArgumentParser()
    parser.add_argument('-e', help = " if specified, use the dictionary you made to classify the document", action="store_true", default = False)
    parser.add_argument('-b', help = " best-result", action="store_true", default = False)
    parser.add_argument('-d', help = " The doc.csv", type = str)
    parser.add_argument('-g', help = " The group.csv", type = str)
    parser.add_argument('-o', help = " output path : *.csv", type = str)
    args = parser.parse_args()

    global N, M, X, K, lamda, theta, maxIteration, p, WordsInGroup, Vocab_List, Save_dir, G

    # Read_Doc(args.d)
    N, M, X , Vocab_List= Read_Doc(args.d)
    print('M = ',M)

    K = 50    # number of topic
    maxIteration = 100  # iter
    threshold = 10

    # lamda[i, j] : p(zj|di)
    lamda = random([N, K])

    # theta[i, j] : p(wj|zi)
    theta = random([K, M])

    # p[i, j, k] : p(zk|di,wj)
    p = zeros([N, M, K])
    lamda, theta = Initial(lamda, theta)
    # EM algo
    oldLoglikelihood = 1
    newLoglikelihood = 1
    print('===== EM Algorithm Start =====\n')
    for i in range(0, maxIteration):
        p = E_Step(theta, lamda, p)
        theta, lamda = M_Step(theta, lamda, p, X)
        newLoglikelihood = LogLikelihood(theta, lamda, X)
        print("[", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "] ", i+1, " iteration  ", str(newLoglikelihood))
        # if(oldLoglikelihood != 1 and newLoglikelihood - oldLoglikelihood < threshold):
        #     break
        oldLoglikelihood = newLoglikelihood
    
    print('\n===== EM Algorithm end =====')

    WordsInGroup ,G = Group(args.g, args.e)
    Save_dir = 'K_'+str(K)+'Iter_'+str(maxIteration)+'Thre_'+str(threshold)

    # OutPut() # save two matrix


    # word_count_classifer(args.o)
    plsa_classifer(args.o)

    print("\n[", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "] OutPut Done!!! " )
  

if __name__ == '__main__':
    main()