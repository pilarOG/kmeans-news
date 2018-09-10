# -*- coding: utf-8 -*-

from gensim.models import Word2Vec
import codecs
import unicodedata
import pickle
import pandas as pd
from pattern.text.es import parse

db = pd.read_csv('RANKING_DOCUMENTOS.tsv', header = 0, delimiter = "\t", encoding = 'utf-8')
#test = pd.read_csv('head_test_v1.csv', header = 0, delimiter = ",", encoding = 'utf-8')
# synonyms = pickle.load(open("../synonyms.pckl", "rb"))

def removeDiacritics(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text).lower().encode('ascii', 'ignore')).replace('¡','').replace('¿','').replace('!','').replace('?','').replace(',','').replace('.','').replace('-','').replace('aaa','a').replace('eee','e').replace('iii','i').replace('ooo','o').replace('uuu','u').replace('   ',' ').replace('  ',' ')



### TODO: AGREGAR LEMMATIZATION



# CORPUS:
# definiciones nodos
ontologia = pd.read_csv('nodos.csv', header = 0, delimiter = ",", encoding = 'utf-8')
corpus = list(ontologia[u'Definición_1'])
corpus += list(ontologia[u'Definición_2'])
corpus += list(ontologia[u'Definición_3'])

#corpus = codecs.open('spanish_billion_words_00', 'r', encoding='utf-8').read().split('\n')
#corpus += codecs.open('spanish_billion_words_01', 'r', encoding='utf-8').read().split('\n')
#corpus += codecs.open('spanish_billion_words_02', 'r', encoding='utf-8').read().split('\n')
#corpus += codecs.open('spanish_billion_words_03', 'r', encoding='utf-8').read().split('\n')
#corpus += codecs.open('spanish_billion_words_04', 'r', encoding='utf-8').read().split('\n')


# artículos
corpus += list(set(list(db[u'RESPUESTA'])))
corpus += list(set(list(db[u'PREGUNTA'])))
corpus += list(set(list(db[u'CANONICA'])))

# libros de ciencia

corpus += codecs.open('Ciencias.txt', 'r', encoding='utf-8').read().split('\n')

## PRE PROCESSING: word embedding

# This is the formant sentences must have
#sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

v = {}
pFunc = [u'del', u'la', u'de', u'y', u'en', u'un', u'el', u'la', u'un', u'una', u'los',
     u'ls', u'las', u'unos', u'unas', u'uns', u'del', u'dl', u'al', u'la', u'el',
     u'las', u'los', u'y', u'con', u'de', u'para', u'por', u'al', u'a', u'como',
     u'desde', u'una', u'un', u'o', u'en', u'me', u'y', u'se', u'que', u' ']

corpora = []
for line in corpus:
    if line and type(line) != float:
        line = removeDiacritics(line)
        new_line = []
        [new_line.append(n) for n in line.split(' ') if n not in pFunc]
    corpora.append(new_line)
corpora.append(['<unk>'])

print corpora


model = Word2Vec(size=150, window=5, min_count=1)
model.build_vocab(corpora)
model.train(corpora, total_examples=model.corpus_count, epochs=model.iter)
word_vectors = model.wv.vocab

model.save('word2vec_v4')
with open('word2vec_vocab_v4.pckl', 'wb') as handle:
    pickle.dump(word_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)
