import unittest
import csv
import panphon
from panphon.segment import Segment
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
        self.assertEqual(self.t.katakana_to_ipa('コンバンハ'), 'kombaɰ̃ɰa') # TODO this depends on how broad the transcription is

class TestWord(unittest.TestCase):
    def setUp(self):
        ipa = 'sɯika'
        self._ft = panphon.FeatureTable()
        self._t = Transcriber()
        self.w = Word(7976, '西瓜', 'スイカ', '漢', ipa)

    def test_to_str(self):
        self.assertEqual(str(self.w), '7976: 西瓜, スイカ, sɯika', 'Should be ' + str(self.w))

    def test_segments(self):
        a_segment = Segment(['syl', 'son', 'cons', 'cont', 'delrel', 'lat', 'nas', 'strid', 'voi', 'sg', 'cg', 'ant', 'cor', 'distr', 'lab', 'hi', 'lo', 'back', 'round', 'velaric', 'tense', 'long', 'hitone', 'hireg'], ftstr='[+syl, +son, -cons, +cont, -delrel, -lat, -nas, 0strid, +voi, -sg, -cg, 0ant, -cor, 0distr, -lab, -hi, +lo, -back, -round, -velaric, +tense, -long, 0hitone, 0hireg]')
        segments = self.w.segments
        # Segments are objects, so equality is by reference. To confirm two segments are the same we show they have distance 0.
        distance = a_segment - segments[4]
        self.assertEqual(distance, 0, 'Segments not identical')

    def test_feature_vectors(self):
        feature_vectors = self.w.feature_vectors
        a_features = [1, 1, -1, 1, -1, -1, -1, 0, 1, -1, -1, 0, -1, 0, -1, -1, 1, -1, -1, -1, 1, -1, 0, 0]
        self.assertEqual(feature_vectors[4], a_features, 'Feature vectors not identical')

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = BCCWJDataset()

    def test_size(self):
        self.assertEqual(len(self.dataset), OUTPUT_CSV_LENGTH)

    def test_accesses(self):
        # TODO these transcriptions assume a BROAD transcription; denote that somewhere
        first_access = self.dataset[0]
        # should be だ,ダ,da,和
        self.assertEqual(str(first_access), '0: だ, ダ, da')
        self.assertEqual(first_access.origin, '和')

        last_access = self.dataset[OUTPUT_CSV_LENGTH - 1]
        # should be ワーピア,ワーピア,ɰaːpia,固
        self.assertEqual(str(last_access), '36395: ワーピア, ワーピア, ɰaːpia')
        self.assertEqual(last_access.origin, '固')

        middle_access_1 = self.dataset[17511]
        # should be 撃沈,ゲキチン,ɡekit͡ɕiɴ,漢
        self.assertEqual(str(middle_access_1), '17511: 撃沈, ゲキチン, ɡekit͡ɕiɴ')
        self.assertEqual(middle_access_1.origin, '漢')

        middle_access_2 = self.dataset[8928]
        # should be ノック,ノック,nokkɯ,外
        self.assertEqual(str(middle_access_2), '8928: ノック, ノック, nokkɯ')
        self.assertEqual(middle_access_2.origin, '外')


if __name__ == '__main__':
    unittest.main()