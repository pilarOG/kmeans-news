import pysptk
import numpy as np

def extract_mfccs(waveforms, settings):
    all_mfccs = []
    padded_mfccs = []
    all_lengths = []
    for waveform in waveforms[:5]:
        mfccs = []
        fs = waveform[0]
        x = waveform[1]
        frame_length = 1024
        frame_step = 512
        for pos in range(0, (len(x)-frame_length), frame_step):
            xw = x[pos:pos+frame_length] * pysptk.blackman(frame_length)
            ext_mfcc = pysptk.sptk.mfcc(xw, fs=fs, order=12)
            mfccs.append(ext_mfcc)

        # Accumulate sentence level mfccs and length
        all_lengths.append(len(mfccs))
        all_mfccs.append(mfccs)

    print all_lengths # To check upsample_alignment
    max_T = max(all_lengths)
    # Do padding given max_T
    padded_mfccs = [np.pad(np.array(m), ((0, max_T-np.array(m).shape[0]), (0, 0)), 'constant') for m in all_mfccs]
    return padded_mfccs
