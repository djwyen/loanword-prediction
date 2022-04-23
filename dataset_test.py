import unittest
import csv
from data_processing.bccwj_dataset import BCCWJDataset
from data_processing.word import Word
from data_processing.transcriber import Transcriber

PATH_TO_OUTPUT_CSV = "data/BCCWJ/pared_BCCWJ.csv"
OUTPUT_CSV_LENGTH = 36396 # as of last generation


class TestTranscriber(unittest.TestCase):
    # partition:
    # want to cover:
    #   glide substitution
    #   geminate consonant realization
    #   long vowels
    #   syllable final nasal place assimilation
    #   affricate joining
    #   yotsugana realization word-initially and intervocalically
    def setUp(self):
        self.t = Transcriber()
    
    # def test_kana_to_intermediate(self):
    #     pass

    def test_katakana_to_ipa(self):
        self.assertEqual(self.t.katakana_to_ipa('オトナ'), 'otona')
        self.assertEqual(self.t.katakana_to_ipa('インターネット'), 'intaːnetto') # tests geminate consonant
        self.assertEqual(self.t.katakana_to_ipa('アズキイリ'), 'azɯkiːɾi') # tests yotsugana intervocalically, long vowels
        self.assertEqual(self.t.katakana_to_ipa('タニン'), 'taɲiɴ') # tests nasal place assimilation
        self.assertEqual(self.t.katakana_to_ipa('キョウシツ'), 'kʲoːɕit͡sɯ') # tests glide substitution, affricate joining
        self.assertEqual(self.t.katakana_to_ipa('ジコウ'), 'd͡ʑikoː') # tests word initial yotsugana
        self.assertEqual(self.t.katakana_to_ipa('コンバンハ'), 'kombaũ͍ɰa')

class TestWord(unittest.TestCase):
    def setUp(self):
        ipa = 'sɯika'
        t = Transcriber()
        self.w = Word(7976, '西瓜', 'スイカ', '漢', ipa, t.ipa_to_numpy_array(ipa))

    def test_to_str(self):
        self.assertEqual(str(self.w), '7976: 西瓜, スイカ, sɯika', 'Should be ' + str(self.w))
    
    def test_pretty_segs(self):
        pretty_segs = [x for x in self.w.pretty_segs()]
        a_features = [1, 1, -1, 1, -1, -1, -1, 0, 1, -1, -1, 0, -1, 0, -1, -1, 1, -1, -1, -1, 1, -1, 0, 0]
        self.assertEqual(pretty_segs[0][0], 's')
        self.assertEqual(pretty_segs[4][1], a_features)

# class TestDataset(unittest.TestCase):
#     def setUp(self):
#         self.dataset = BCCWJDataset()

#     def test_size(self):
#         self.assertEqual(len(self.dataset), OUTPUT_CSV_LENGTH)

#     def test_accesses(self):
#         # TODO wait for BCCWJDataset to change its __getitem__ to return Words before filling this out
#         first_access = self.dataset[0]
#         # should be だ,ダ,da,和
#         self.assertEqual(first_access)

#         last_access = self.dataset[OUTPUT_CSV_LENGTH - 1]
#         # should be ワーピア,ワーピア,ɰaːpia,固

#         middle_access_1 = self.dataset[17512]
#         # should be 撃沈,ゲキチン,ɡekit͡ɕiɴ,漢

#         middle_access_2 = self.dataset[8928]
#         # should be ノック,ノック,nokkɯ,外





if __name__ == '__main__':
    unittest.main()