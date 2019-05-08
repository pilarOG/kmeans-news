# Train a model to classify each frame as vowel/not
# After classification find the peak intensity on consecutive frames classified as vowels
# Add marks at the chosen points and calculate speech rate using them


# 1) Where do we get data from? would it need to be of the same language?
# I only have pre-aligned data for spanish, so I will train with that and I will test in English
# 2) What model?
# I would like to start trying with LSTMs
# 3) What features?
# Let's start with mfccs

from load_data import load_waves, load_labels
from configure import load_config
from argparse import ArgumentParser
from extract_features import extract_mfccs
import numpy as np


### MAIN ###

a = ArgumentParser()
a.add_argument('-c', dest='config', required=True, type=str)
opts = a.parse_args()
settings = load_config(opts.config)
waveforms = load_waves(settings)
mfccs = extract_mfccs(waveforms, settings)
# print mfccs.shape # (number of files, max_T, 12 coeff)
alignments = load_labels(settings)
