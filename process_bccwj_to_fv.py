"""
Takes in the processed bccwj words and creates a huge csv with all the words represented as vectors.
Created because I realized that panphon transcriptions took up significant overhead in using
a PyTorch dataloader and made my model unable to train at any reasonable rate.
"""
import csv
import pandas as pd
import numpy as np
from data_processing.transcriber import Transcriber

PATH_TO_PROCESSED_CSV = "data/BCCWJ/pared_BCCWJ.csv"
PATH_TO_FV_GZ = "data/BCCWJ/fv_pared_BCCWJ.gz"

MAX_SEQ_LEN_NO_PAD = 20
NUM_OF_PANPHON_FEATURES = 24
PAD_FV = [0] * NUM_OF_PANPHON_FEATURES
END_FV = [2] * NUM_OF_PANPHON_FEATURES

def length_of_ipa(ipa):
    '''
    Quick helper function to compute the length of an ipa string by removing extraneous segments
    '''
    return len(ipa) - ipa.count('ː') - ipa.count('ʲ') - ipa.count('ç') - (2*ipa.count('d͡ʑ')) - (2*ipa.count('d͡z')) - (2*ipa.count('t͡ɕ')) - (2*ipa.count('t͡s')) - ipa.count('ɰ̃') - ipa.count('ĩ')

def main():
    # the huge array that will store a representation of each word
    word_array = []

    t = Transcriber()
    with open(PATH_TO_PROCESSED_CSV) as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        
        for i, line in enumerate(reader):
            ipa = line[2]
            shorthand = t.ipa_to_shorthand(ipa)
            fv = t.shorthand_to_fv(shorthand)

            length_diff = MAX_SEQ_LEN_NO_PAD - length_of_ipa(ipa)
            fv.append(END_FV)
            for _ in range(length_diff):
                fv.append(PAD_FV)

            flat_fv = []
            for vec in fv:
                flat_fv.extend(vec)
            
            word_array.append(flat_fv)
    
    word_array = np.array(word_array)
    np.savetxt(PATH_TO_FV_GZ, word_array)

if __name__ == '__main__':
    main()