import numpy as np
import torch
from panphon import FeatureTable
from panphon.segment import Segment

from .transcriber import Transcriber
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
        fv = self._t.ipa_to_feature_vectors(ipa)
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

    def loan_word(self, ipa):
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
