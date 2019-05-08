import os
from scipy.io import wavfile
import codecs

def load_waves(settings):
    # List of (fs, x) tuples
    wavefiles = [wavfile.read(settings.wav_folder+'/'+filepath) for filepath in os.listdir(settings.wav_folder)]
    return wavefiles

def load_labels(settings):
    alignments = [codecs.open(settings.label_folder+'/'+filepath, encoding='utf-8').read().split('\n') for filepath in os.listdir(settings.label_folder)]
    return alignments
