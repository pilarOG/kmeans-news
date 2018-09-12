# pip3 install wikipedia-api
# https://github.com/martin-majlis/Wikipedia-API/

import unicodedata

def removeDiacriticsStopw(text):
    new_text = []
    stopw = [u'del', u'la', u'de', u'y', u'en', u'un', u'el', u'la', u'un', u'una', u'los',
             u'ls', u'las', u'unos', u'unas', u'uns', u'del', u'dl', u'al', u'la', u'el',
             u'las', u'los', u'y', u'con', u'de', u'para', u'por', u'al', u'a',
             u'desde', u'una', u'un', u'o', u'en', u'me', u'y', u'se', u'que', u'como', u'porque']
    [new_text.append(w) for w in text.split(' ') if w not in stopw]
    return new_text

input = 'cuando falleció'

import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia(
        language='es',
        extract_format=wikipediaapi.ExtractFormat.WIKI)

p_wiki = wiki_wiki.page("Elena Caffarena")

# we could sotre info from different sections of the wikipedia page
info = p_wiki.text
info = [l.lower() for l in info.replace('.', '\n').split('\n') if l.replace(' ', '')]

set_input = set(input.split(' '))

# METHOD 1: Boolean Search

set_scores = []
for l in info:
    info_line = [n for n in removeDiacriticsStopw(l)]
    set_info = set(info_line)


    inter = float(len(set_info.intersection(set_input)))/float(len(input.split(' '))) # mientras mas grande mejor
    diff = float(len(set_info.symmetric_difference(set_input)))/float(len(input.split(' '))) # mientras mas chico mejor
    score = inter * diff # mientras mas grande mejor
    if score != 0: set_scores.append([score, l])
