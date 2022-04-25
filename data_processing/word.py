"""
Word is a class bundling together several properties of a word in the pared BCCWJ dataset,
namely its original idx in the pared dataset, its lemma (word), its form in kana,
its sublexical origin, its transcription in IPA, and the panphon segments it induces
"""

from .transcriber import Transcriber

class Word():
    # TODO add another form of the segments besides just the vectors
    #      and in particular make it possible to access things by particular features like voicing or whatever
    #      the upshot is, read the panphon documentation...
    def __init__(self, idx, word, kana, origin, ipa):
        self._t = Transcriber()
        self.idx = idx # note that a word's ID will not necessarily be its key in the BCCWJDataset due to shuffling
        self.word = word
        self.kana = kana
        self.origin = origin
        self.ipa = ipa
    
    def __str__(self):
        return f'{self.idx}: {self.word}, {self.kana}, {self.ipa}'
    
    def segments(self):
        return self._t.ipa_to_panphon_word(self.ipa)
    
    def vectors(self):
        return self._t.ipa_to_numpy_array(self.ipa)
