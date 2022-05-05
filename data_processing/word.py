"""
Word is a class bundling together the properties of a word in the pared BCCWJ dataset,
namely its original idx in the pared dataset, its lemma (word), its form in kana,
its sublexical origin, and its transcription in IPA.
"""

class Word():
    # TODO add another form of the segments besides just the vectors
    #      and in particular make it possible to access things by particular features like voicing or whatever
    #      the upshot is, read the panphon documentation...
    def __init__(self, idx, word, kana, origin, ipa):
        self.idx = idx # note that a word's ID will not necessarily be its key in the BCCWJDataset due to shuffling
        self.word = word
        self.kana = kana
        self.origin = origin
        self.ipa = ipa
    
    def __str__(self):
        return f'{self.idx}: {self.word}, {self.kana}, {self.ipa}'
