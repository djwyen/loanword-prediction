import csv

import numpy as np
import torch
import torch.nn.functional as F
from panphon import FeatureTable
from panphon.segment import Segment

from .transcriber import Transcriber, FEATURE_WEIGHTS, SHORTHAND_TO_FV_CSV
from .bccwj_dataset import MAX_SEQ_LEN_NO_PAD, NUM_PHONETIC_FEATURES, CATEGORIES_PER_FEATURE, PAD_BINARY_FV, END_BINARY_FV

class WordLoaner():
    '''
    Class that wraps a trained AutoEncoder model to provide a seamless interface for
    "loaning" words with this model
    '''
    def __init__(self, model):
        self.model = model
        self._t = Transcriber()
        self._ft = FeatureTable()

        self.FEATURES = ['syl', 'son', 'cons', 'cont', 'delrel', 'lat', 'nas', 'strid', 'voi', 'sg', 'cg', 'ant', 'cor', 'distr', 'lab', 'hi', 'lo', 'back', 'round', 'velaric', 'tense', 'long']
        self.shorthand_to_fv_dict = {}
        with open(SHORTHAND_TO_FV_CSV) as f:
            reader = csv.reader(f)
            header = next(reader)
            for line in reader:
                shorthand_char = line[0]
                ipa = line[1]
                fv_as_strings = line[2:]
                fv = [int(s) for s in fv_as_strings]
                self.shorthand_to_fv_dict[shorthand_char] = fv

    def _pad_word(self, fv):
        # pads a word's feature vector to be a correctly formatted input to the model
        length_diff = MAX_SEQ_LEN_NO_PAD - len(fv)
        fv.append(END_BINARY_FV)
        for _ in range(length_diff):
            fv.append(PAD_BINARY_FV)
        return fv

    def word_to_fv(self, ipa, discretize=True):
        # returns the predicted features for a given word, given in ipa.
        # ie turns a word into feature vectors.
        # if discretize is true, the outputs are rounded to the nearest integer

        # assert(self._ft.validate_word(ipa))
        # fv = self._t.ipa_to_feature_vectors(ipa)
        shorthand = self._t.ipa_to_shorthand(ipa)
        fv = self._t.shorthand_to_fv(shorthand)
        fv = self._t.fv_to_binary_fv(fv)
        fv = self._pad_word(fv)
        fv = torch.tensor(np.array(fv)) # fv: (L, H_in)
        fv = fv.unsqueeze(0) # fv: (1, L, H_in)
        fv = self.model.loan_word_from_fv(fv) # fv: (1, 2*L, H_in)
        fv = fv.squeeze() # fv: (2*L, H_in)
        # convert to a numpy array for convenience
        fv = fv.numpy()
        if discretize:
            fv = np.rint(fv)

        return fv

    def greedy_select_segment(self, fv, weighted=True):
        # returns the ipa character in Japanese with the closest fv representation
        # to the given segment
        least_distance = None
        closest_seg = None
        # extend shorthand dict with vectors for the eow and pad tokens
        # we need to binarize the fv
        shorthand_to_fv_dict_extended = {k: self._t.fv_to_binary_fv([v])[0] for k, v in self._t.shorthand_to_fv_dict.items()}
        # shorthand_to_fv_dict_extended['$'] = [0] * NUM_PHONETIC_FEATURES
        # shorthand_to_fv_dict_extended['_'] = [1] * NUM_PHONETIC_FEATURES

        # print('for', fv)
        for c, vec in shorthand_to_fv_dict_extended.items():
            # print(f'considering {c}')
            # print(vec)
            total = 0
            for val1, val2, w, feature in zip(fv, vec, FEATURE_WEIGHTS, self.FEATURES):
                if not weighted:
                    w = 1
                diff = w * (np.abs(val1 - val2))
                # print(f'diff at {feature}: {diff}')
                total += diff
            if least_distance is None or total < least_distance:
                least_distance = total
                closest_seg = c
            # print('total distance', total)
        shorthand_to_ipa = {v: k for k, v in self._t.ipa_to_shorthand_dict.items()}
        # shorthand_to_ipa['$'] = '.'
        # shorthand_to_ipa['_'] = ''
        # print('---')
        return shorthand_to_ipa[closest_seg]

    def fv_to_closest_word(self, fvs):
        # takes in a list of feature vectors and returns the IPA string with the closest segment for each feature vector
        result = ''
        for fv in fvs:
            result += self.greedy_select_segment(fv, weighted=True)
        return result

    def loan_word(self, ipa):
        # TODO DEPRECATED
        # turns an ipa word into its loaned form and returns the segments
        permissible_values = {1, 0, -1}
        
        word_fv = self.word_to_fv(ipa, discretize=True)

        closest_transcription = ''
        for seg_fv in word_fv:
            # skip pad or eow tokens
            # print(seg_fv)
            closest_transcription += self._t.greedy_select_segment(seg_fv, weighted=True)

        # segments = []

        # for segment_fv in word_fv:
        #     # assert(len(segment_fv) == NUM_FEATURES)
        #     seg_dict = {}
        #     for feature, value in zip(self.FEATURES, segment_fv):
        #         char_feature_sign = ''
        #         # print('val', value)
        #         if value in permissible_values:
        #             seg_dict[feature] = int(value)
        #         else:
        #             print(f'untranscribable value for {feature}: {value}')
        #             break

        #     # assert(len(seg_dict) == NUM_FEATURES)
        #     seg = Segment(self.FEATURES, seg_dict)
        #     segments.append(seg)
        
        return word_fv, closest_transcription
    
    def read_segment(self, seg):
        # returns a more readable string pulling the relevant features of a segment
        # TODO should this be under Transcriber?
        # I like having it here just because we'll only use it here,
        # but conceptually it seems it should belong in Transcriber

        result = ''
        voicing = 'voiced' if seg['voi'] == '+' else 'voiceless'
        nasality = 'nasal' if seg['nas'] == '+' else ''
        high = seg['hi'] == '+'
        low = seg['lo'] == '+'
        back = seg['back'] == '+'
        _round = seg['round'] == '+'


        if seg['con'] == '+':
            # assume it's a consonant
            place = ''
            anterior = seg['ant'] == '+'
            coronal = seg['cor'] == '+'
            labial = seg['lab'] == '+'

            if anterior and coronal:
                place = 'alveolar'
            elif high and back:
                place = 'velar'
            elif labial:
                place = 'labial'

            manner = ''
            continuous = seg['cont'] == '+'
            delayed_release = seg['delrel'] == '+'
            
            if continuous and delayed_release:
                manner = 'fricative'
            elif (not continuous) and delayed_release:
                manner = 'affricate'
            elif (not continuous) and (not delayed_release):
                manner = 'stop'

            result = f'{voicing} {place} {manner}'
        elif seg['con'] == '-':
            # assume it's a vowel

            height = ''

            if (not high) and (not low):
                height = 'mid'
            elif high:
                height = 'high'
            elif low:
                height = 'low'

            backness = 'back' if back else 'front' # unfortunately panphon doesn't let you define 'central' easily

            result = f'{height} {backness} vowel'
        else:
            # can't make strong assumptions...
            result = 'unknown'
        
        return result
