from collections import defaultdict
import glob
import re
import scipy.io
import numpy as np
import pdb
import re


BASE_DIR = '/Users/Dom/Desktop/CS189/hw1/data/'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')

frequency_spam = {}
for filename in spam_filenames:
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        text_string = f.read().lower()
        match_pattern = re.findall(r'\b[a-z]{3,15}\b', text_string)
        for word in match_pattern:
            count = frequency_spam.get(word,0)
            frequency_spam[word] = count + 1
most_frequent = dict(sorted(frequency_spam.items(), key=lambda elem: elem[1], reverse=False))
most_frequent_count = most_frequent.keys()
for words in most_frequent_count:
    print(words, most_frequent[words])

frequency_spam = {}
for filename in ham_filenames:
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        text_string = f.read().lower()
        match_pattern = re.findall(r'\b[a-z]{3,15}\b', text_string)
        for word in match_pattern:
            count = frequency_spam.get(word,0)
            frequency_spam[word] = count + 1
most_frequent = dict(sorted(frequency_spam.items(), key=lambda elem: elem[1], reverse=False))
most_frequent_count = most_frequent.keys()
for words in most_frequent_count:
    print(words, most_frequent[words])