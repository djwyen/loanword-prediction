"""
Takes in the processed bccwj words and creates a huge csv with all the words represented as vectors.
Created because I realized that panphon transcriptions took up significant overhead in using
a PyTorch dataloader and made my model unable to train at any reasonable rate.
"""
import csv
import numpy as np
from data_processing.transcriber import Transcriber
from utils import segment_len_of_ipa

PATH_TO_PROCESSED_CSV = "data/BCCWJ/pared_BCCWJ.csv"
PATH_TO_FV_GZ = "data/BCCWJ/fv_pared_BCCWJ.gz"

MAX_SEQ_LEN_NO_PAD = 20
NUM_PHONETIC_FEATURES = 22 # Panphon by default gives you 24 features, but the last two corresond to tonal features so I drop them
CATEGORIES_PER_FEATURE = 3 # the categories being {+, -, 0} in that order
END_MULTIHOT_FV = [0] * NUM_PHONETIC_FEATURES * CATEGORIES_PER_FEATURE
PAD_MULTIHOT_FV = [1] * NUM_PHONETIC_FEATURES * CATEGORIES_PER_FEATURE

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
            fv = t.fv_to_multihot(fv)

            # add the end-of-word token and then pad to the max seq length
            length_diff = MAX_SEQ_LEN_NO_PAD - segment_len_of_ipa(ipa)
            fv.append(END_MULTIHOT_FV)
            for _ in range(length_diff):
                fv.append(PAD_MULTIHOT_FV)

            # flatten the word by concatenating its feature vectors into one list
            flat_fv = []
            for vec in fv:
                flat_fv.extend(vec)
            
            word_array.append(flat_fv)
    
    word_array = np.array(word_array)
    np.savetxt(PATH_TO_FV_GZ, word_array)

if __name__ == '__main__':
    main()