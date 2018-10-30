# pip3 install wikipedia-api
# https://github.com/martin-majlis/Wikipedia-API/

import unicodedata
from nltk import bigrams, trigrams
from math import log
import re

def removeDiacriticsStopw(text):
    new_text = []
    stopw = [u'del', u'la', u'de', u'y', u'en', u'un', u'el', u'la', u'un', u'una', u'los',
             u'ls', u'las', u'unos', u'unas', u'uns', u'del', u'dl', u'al', u'la', u'el',
             u'las', u'los', u'y', u'con', u'de', u'para', u'por', u'al', u'a',
             u'desde', u'una', u'un', u'o', u'en', u'me', u'y', u'se', u'que', u'como', u'porque']
    [new_text.append(w) for w in text.split(' ') if w not in stopw]
    return new_text


def builLM(text):
    text_trigrams = {}
    text_bigrams = {}
    text_unigrams = {}
    # Build Unigrams
    V = 0
    for n in text:
        if n not in text_unigrams:
            text_unigrams[n] = 2
            V += 2
        else:
            text_unigrams[n] += 1
            V += 1
    # Buil Bigrams
    for n in list(bigrams(text, pad_left=True, pad_right=True)):
        if n not in text_bigrams:text_bigrams[n] = 2
        else: text_bigrams[n] += 1
    # Build Trigram
    for n in list(trigrams(text, pad_left=True, pad_right=True)):
        if n not in text_trigrams:text_trigrams[n] = 2
        else: text_trigrams[n] += 1

    return V, text_trigrams, text_bigrams, text_unigrams

input = 'cuando ganó el premio nobel de literatura'

import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia(
        language='es',
        extract_format=wikipediaapi.ExtractFormat.WIKI)

p_wiki = wiki_wiki.page("Gabriela Mistral")

# we could store info from different sections of the wikipedia page
info = p_wiki.text
info = [l.lower() for l in info.replace('.', '\n').split('\n') if l.replace(' ', '')]


# METHOD 1: Boolean Search

set_scores = []
set_input = set()

for l in info:
    line = [n for n in removeDiacriticsStopw(l)]
    info_line = ' '.join(line)
    for n in line:
        if re.findall('[1-9][0-9]{3}', info_line):
            for i in re.findall('[1-9][0-9]{3}', info_line):
                info_line.replace(i, 'cuando')
    info_line = info_line.split(' ')
    set_info = set(info_line)
    inter = float(len(set_info.intersection(set_input)))/float(len(input.split(' '))) # mientras mas grande mejor
    diff = float(len(set_info.symmetric_difference(set_input)))/float(len(input.split(' '))) # mientras mas chico mejor
    score = inter * diff # mientras mas grande mejor
    if score != 0: set_scores.append([score, l])

print (set_scores)

# METHOD 2: LM

lm_scores = []
input_words = input.split(' ')
big = list(bigrams(input_words, pad_left=True, pad_right=True))
trig = list(trigrams(input_words, pad_left=True, pad_right=True))

for l in info:
    info_line = [n for n in removeDiacriticsStopw(l)]
    V, resp_trigrams, resp_bigrams, resp_unigrams = builLM(info_line)

    # Calculate probability
    probs, bc, tc = 0, 0, 0
    for i in range(0, len(big)):
        if big[i] in resp_bigrams: bc = resp_bigrams[big[i]]
        else: bc = 1
        if trig[i] in resp_trigrams: tc = resp_trigrams[trig[i]]
        else: tc = 1
        probs += log(float(tc)/float(bc))
    if probs != 0.0: lm_scores.append([probs, l])

print (lm_scores)
