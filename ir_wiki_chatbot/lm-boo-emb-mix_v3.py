# -*- coding: utf-8 -*-

from gensim.models import Word2Vec
from nltk import bigrams, trigrams
import pandas as pd
import pickle
from pattern.text.es import parse
from math import log
import unicodedata

emb_v = pickle.load(open('word2vec_v.pckl', 'rb'))

class Input():
    def __init__(self, input, arguments, tipo):
        self.original_input = input
        self.input_no_stopw_no_diac = removeDiacriticsStopw(input)
        self.input_lemmas = lemmatize(self.input_no_stopw_no_diac)

        self.original_arguments = arguments.replace(', ',' ').replace('[','').replace(']','').split(' ')
        arguments_no_stopw_no_diac = []
        [arguments_no_stopw_no_diac.append(removeDiacriticsStopw(a)) for a in self.original_arguments]
        self.arguments_no_stopw_no_diac = arguments_no_stopw_no_diac
        arguments_lemmas = []
        for a in self.arguments_no_stopw_no_diac:
            [arguments_lemmas.append(i) for i in lemmatize(a) ]
        self.arguments_lemmas = arguments_lemmas

        self.tipo = tipo


class Document():
    def __init__(self, canonica, variante, respuesta, nodos, categoria):
        self.original_canonica = canonica
        self.canonica_no_stopw_no_diac = removeDiacriticsStopw(canonica)
        self.canonica_lemmas = lemmatize(self.canonica_no_stopw_no_diac)

        self.original_variante = variante
        self.variante_no_stopw_no_diac = removeDiacriticsStopw(variante)
        self.variante_lemmas = lemmatize(self.variante_no_stopw_no_diac)

        self.original_respuesta = respuesta
        self.respuesta_no_stopw_no_diac = removeDiacriticsStopw(respuesta)
        self.respuesta_lemmas = lemmatize(self.respuesta_no_stopw_no_diac)

        respuesta_sentences = []
        [respuesta_sentences.append(s) for s in respuesta.split('.')]
        self.respuesta_sentences = respuesta_sentences

        respuesta_sentences_no_stopw_no_diac = []
        [respuesta_sentences_no_stopw_no_diac.append(removeDiacriticsStopw(s)) for s in self.respuesta_sentences]
        self.respuesta_sentences_no_stopw_no_diac = respuesta_sentences_no_stopw_no_diac

        self.original_nodos = nodos.replace(']', '').replace('[','').split(', ')
        nodos_no_stopw_no_diac = []
        [nodos_no_stopw_no_diac.append(removeDiacriticsStopw(n)) for n in self.original_nodos]
        self.nodos_no_stopw_no_diac = nodos_no_stopw_no_diac
        nodos_lemmas = []
        for n in nodos_no_stopw_no_diac:
            [nodos_lemmas.append(i) for i in lemmatize(n)]
        self.nodos_lemmas = nodos_lemmas

        self.categoria = categoria


class Analysis():
    def __init__(self):
        self.lm_respuesta_input = []
        self.bool_variante_input = []
        self.bool_argumento_nodo = []
        self.emb_argumento_nodo = []
        self.emb_input_variante = []
        self.emb_input_respuesta_passage = []

        self.all_canonicas = []
        self.all_variantes = []
        self.all_input = []
        self.all_nodos = []
        self.all_argumentos = []
        self.all_respuestas = []

        self.label = []


def removeDiacriticsStopw(text):
    new_text = []
    text = ''.join(c for c in unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')).lower().replace(',', '').replace('.', '').replace('(', '').replace(')', '').replace(':', '').replace('<new line>', '')
    stopw = [u'del', u'la', u'de', u'y', u'en', u'un', u'el', u'la', u'un', u'una', u'los',
             u'ls', u'las', u'unos', u'unas', u'uns', u'del', u'dl', u'al', u'la', u'el',
             u'las', u'los', u'y', u'con', u'de', u'para', u'por', u'al', u'a',
             u'desde', u'una', u'un', u'o', u'en', u'me', u'y', u'se', u'que', u'como', u'porque']
    [new_text.append(w) for w in text.split(' ') if w not in stopw]
    return new_text


def lemmatize(text):
    new_text = []
    parsed = parse(' '.join(text), lemmata=True)
    [new_text.append(item.split('/')[-1]) for item in parsed.split(' ')]
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

def inEmbedding(text):
    new_text = []
    for t in text: # already has synonyms
        if emb_v.get(t):
            if emb_v[t] != 1: new_text.append(t)
            else: new_text.append('<unk>')
        else:
            new_text.append('<unk>')
    return new_text

### MAIN ###

#bd = pd.read_csv('RANKING_DOCUMENTOS.tsv', header = 0, delimiter = "\t", encoding = 'utf-8')
synonyms = pickle.load(open("synonyms.pckl", "rb"))
data = pd.read_csv('labeled_data.csv', header = 0, delimiter = ",", encoding = 'utf-8')
model = Word2Vec.load('word2vec')
emb_v = pickle.load(open('word2vec_v.pckl', 'rb'))

analysisObject = Analysis()
for j in range(0, len(data['inputs'])):
    print data['inputs'][j]
    inputObject = Input(data['inputs'][j], data['args'][j], data['tipo_pregunta'][j])
    analysisObject.label.append(data['label'][j])
    #for i in range(0, len(bd['RANKING_CATEGORIAS_ID'])):
    docObject = Document(data['canonicas'][j], data['variante'][j], data['respuesta'][j], data['nodos'][j], data['tipo_pregunta'][j])


    ### LANGUAGE MODEL APPROACH

    # Answer compared to input
    # Process Answer
    V, resp_trigrams, resp_bigrams, resp_unigrams = builLM(docObject.respuesta_lemmas)
    # Process Input
    # Check if all words exist in unigram LM, if not, check synonyms, if not, assign a very small account
    for n in range(0, len(inputObject.input_lemmas)):
        if inputObject.input_lemmas[n] not in resp_unigrams:
            if inputObject.input_lemmas[n] in synonyms:
                for m in synonyms[inputObject.input_lemmas[n]][0][0]:
                    if m in resp_unigrams:
                        inputObject.input_lemmas[n] = removeDiacriticsStopw(m)[0]
    # Input bigrams and trigrams
    big = list(bigrams(inputObject.input_lemmas, pad_left=True, pad_right=True))
    trig = list(trigrams(inputObject.input_lemmas, pad_left=True, pad_right=True))
    # Calculate probability
    probs, bc, tc = 0, 0, 0
    for i in range(0, len(big)):
        if big[i] in resp_bigrams: bc = resp_bigrams[big[i]]
        else: bc = 1
        if trig[i] in resp_trigrams: tc = resp_trigrams[trig[i]]
        else: tc = 1
        probs += log(float(tc)/float(bc))
    analysisObject.lm_respuesta_input.append(probs)

    ### BOOLEAN APPROACH

    # Variant compared to input
    # Process variante
    v_L = len(docObject.variante_lemmas)
    var_set = set(docObject.variante_lemmas)
    # Process Input
    for n in range(0, len(inputObject.input_lemmas)):
        if inputObject.input_lemmas[n] not in var_set:
            if inputObject.input_lemmas[n] in synonyms:
                for m in synonyms[inputObject.input_lemmas[n]][0][0]:
                    if m in var_set:
                        inputObject.input_lemmas[n] = removeDiacriticsStopw(m)[0]
    i_L = len(inputObject.input_lemmas)
    inp_set = set(inputObject.input_lemmas)
    # Compare sets
    v_i_i = float(len(inp_set.intersection(var_set)))/float(v_L) # mientras mas grande mejor
    v_i_d = float(len(inp_set.symmetric_difference(var_set)))/float(v_L) # mientras mas chico mejor
    v_i_t = v_i_i * v_i_d # mientras mas grande mejor
    analysisObject.bool_variante_input.append(v_i_t)

    # Arguments compared to nodes
    # Find synonyms
    nod_set = set(docObject.nodos_lemmas)
    a_set = set(inputObject.arguments_lemmas)
    args_set = []
    for n in a_set:
        if n not in nod_set:
            if n in synonyms:
                for m in synonyms[n][0][0]:
                    if m in nod_set:
                        args_set.append(removeDiacriticsStopw(m)[0])
                    else: args_set.append(n)
            else: args_set.append(n)
        else: args_set.append(n)
    args_set = set(args_set)
    n_a_i = len(args_set.intersection(nod_set))
    analysisObject.bool_argumento_nodo.append(n_a_i)

    ### EMBEDDING APPROACH

    # Arguments compared to nodes
    # Process argument
    args_emb = inEmbedding(args_set)

    # Process node
    nod_emb = inEmbedding(docObject.nodos_lemmas)

    # Compare each word in the args with each word in the nodes
    a_n_e = []
    for x in args_emb:
        for y in nod_emb:
            d = model.wv.n_similarity([x], [y])
            a_n_e.append(d)
    distance = sorted(a_n_e)[-1]
    analysisObject.emb_argumento_nodo.append(distance)

    # Input compared to variant, no using lemmas
    # Process input
    input_emb = inEmbedding(inputObject.input_no_stopw_no_diac)
    # Process variant
    var_emb = inEmbedding(docObject.variante_no_stopw_no_diac)
    # Compare
    distance = model.wv.n_similarity(input_emb, var_emb)
    analysisObject.emb_input_variante.append(distance)

    # Sentences in answer compared to input, no lemmas
    s_i_e = []
    sent_emb = []
    for sentence in docObject.respuesta_sentences_no_stopw_no_diac:
        sent_emb = inEmbedding(sentence)
        distance = model.wv.n_similarity(input_emb, sent_emb)
        s_i_e.append(distance)
    distance = sorted(s_i_e)[-1]
    analysisObject.emb_input_respuesta_passage.append(distance)

    # Save all data
    analysisObject.all_canonicas.append(docObject.original_canonica)
    analysisObject.all_variantes.append(docObject.original_variante)
    analysisObject.all_input.append(inputObject.original_input)
    analysisObject.all_nodos.append(docObject.original_nodos)
    analysisObject.all_argumentos.append(inputObject.original_arguments)
    analysisObject.all_respuestas.append(docObject.original_respuesta)





output = pd.DataFrame(data = {"inputs":analysisObject.all_input,
                              "variante":analysisObject.all_variantes,
                              "nodos": analysisObject.all_nodos,
                              "args": analysisObject.all_argumentos,
                              "respuesta":analysisObject.all_respuestas,
                              "canonicas":analysisObject.all_canonicas,
                              "lm_score":analysisObject.lm_respuesta_input,
                              "bool_v_i":analysisObject.bool_variante_input,
                              "bool_a_n":analysisObject.bool_argumento_nodo,
                              "emb_a_n":analysisObject.emb_argumento_nodo,
                              "emb_i_v":analysisObject.emb_input_variante,
                              "emb_i_r":analysisObject.emb_input_respuesta_passage,
                              "label":analysisObject.label})

output.to_csv('resultados_lm_boo_emb_mixed_v5.csv', sep=",", encoding = 'utf-8',decimal=",")
