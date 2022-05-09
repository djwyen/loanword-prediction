import unittest
import csv
import numpy as np
import panphon
from panphon.segment import Segment
from data_processing.bccwj_dataset import BCCWJDataset
from data_processing.word import Word
from data_processing.transcriber import Transcriber

OUTPUT_CSV_LENGTH = 36396 # as of last generation
MAX_SEQ_LEN_NO_PAD = 20
MAX_SEQ_LEN_WITH_PAD = 21
N_FEATURES = 24


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
        segments = self._t.ipa_to_panphon_segments(self.w.ipa)
        # Segments are objects, so equality is by reference. To confirm two segments are the same we show they have distance 0.
        distance = a_segment - segments[4]
        self.assertEqual(distance, 0, 'Segments not identical')

    def test_feature_vectors(self):
        feature_vectors = self._t.ipa_to_feature_vectors(self.w.ipa)
        a_features = [1, 1, -1, 1, -1, -1, -1, 0, 1, -1, -1, 0, -1, 0, -1, -1, 1, -1, -1, -1, 1, -1, 0, 0]
        self.assertEqual(feature_vectors[4], a_features, 'Feature vectors not identical')

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = BCCWJDataset(range(OUTPUT_CSV_LENGTH), MAX_SEQ_LEN_NO_PAD)

    def test_size(self):
        self.assertEqual(len(self.dataset), OUTPUT_CSV_LENGTH)
    
    def test_output_dims(self):
        entry = next(iter(self.dataset))
        self.assertIsInstance(entry, np.ndarray)
    
    def test_accesses(self):
        first_access = self.dataset[0]
        self.assertEqual(first_access.shape, (MAX_SEQ_LEN_WITH_PAD, N_FEATURES))
        self.assertTrue(all([(lambda x: x == 2) for x in first_access[-1, :]])) # ie the EOW token

        middle_access = self.dataset[17511]
        self.assertEqual(middle_access.shape, (MAX_SEQ_LEN_WITH_PAD, N_FEATURES))
        self.assertTrue(all([(lambda x: x == 2) for x in middle_access[-1, :]])) # ie the EOW token


if __name__ == '__main__':
    unittest.main()