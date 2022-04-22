"""
Word is a class bundling together several properties of a word in the pared BCCWJ dataset,
namely its original idx in the pared dataset, its lemma (word), its form in kana,
its sublexical origin, its transcription in IPA, and the panphon segments it induces
"""

class Word():
    def __init__(self, idx, word, kana, origin, ipa, segments):
        self.idx = idx # note that a word's ID will not necessarily be its key in the BCCWJDataset due to shuffling
        self.word = word
        self.kana = kana
        self.origin = origin
        self.ipa = ipa
        self.segments = segments
    
    def __str__(self):
        return f'{self.idx}: {self.word}, {self.kana}, {self.ipa}'
    
    def prettySegs(self):
        return zip(self.ipa, self.segments)
