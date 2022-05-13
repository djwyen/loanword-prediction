import csv
from typing import List, Mapping
from .data_errors import UnsupportedKanaError

import numpy as np
import panphon
from panphon.segment import Segment

KATAKANA_TO_IPA_CSV = 'kana_to_ipa.csv'
SHORTHAND_TO_FV_CSV = 'shorthand_to_fv.csv'

# represents how strict to be with transcriptions, currently just two settings
class TranscriptionStyle:
    BROAD = 1
    NARROW = 2

TRANSCRIPTION_STYLE = TranscriptionStyle.BROAD # can be set via a commandline argument, eventually

NUM_PHONETIC_FEATURES = 22 # Panphon by default gives you 24 features, but the last two corresond to tonal features so I drop them
CATEGORIES_PER_FEATURE = 3 # the categories being {+, -, 0} in that order

# pulled from https://github.com/dmort27/panphon/blob/master/panphon/data/feature_weights.csv
# some arbitrary weighting of the features
FEATURE_WEIGHTS = [1,1,1,0.5,0.25,0.25,0.25,0.125,0.125,0.125,0.125,0.25,0.25,0.125,0.25,0.25,0.25,0.25,0.25,0.25,0.125,0.25]
TRIPLICATE_FEATURE_WEIGHTS = []
for w in FEATURE_WEIGHTS:
    TRIPLICATE_FEATURE_WEIGHTS.extend([w, w, w])

class Transcriber():
    """
    A class to wrap methods for converting Japanese words between scripts:
    - Kana to IPA
    - Kana to Romaji TODO
    - IPA to feature vectors (via panphon)
    - IPA to panphon Segments

    Currently there is only support for Katakana, which is what BCCWJ is written in. TODO support hiragana
    """
    def __init__(self, style = TranscriptionStyle.BROAD):
        self.style = style
        self.katakana_to_intermediate = {}
        with open(KATAKANA_TO_IPA_CSV) as f:
            reader = csv.reader(f)
            header = next(reader)
            for line in reader:
                kana, ipa = line
                self.katakana_to_intermediate[kana] = ipa
        self.ipa_to_shorthand_dict = {}
        self.ipa_sub_order = [] # the order in which to do the substitutions
        self.shorthand_to_fv_dict = {}
        with open(SHORTHAND_TO_FV_CSV) as f:
            reader = csv.reader(f)
            header = next(reader)
            for line in reader:
                shorthand_char = line[0]
                ipa = line[1]
                fv_as_strings = line[2:]
                fv = [int(s) for s in fv_as_strings]

                self.ipa_to_shorthand_dict[ipa] = shorthand_char
                self.ipa_sub_order.append(ipa)
                self.shorthand_to_fv_dict[shorthand_char] = fv

    def katakana_to_ipa(self, word_in_katakana):
        """
        Converts a word or series of words in katakana to its IPA equivalent.
        
        The transcription is pretty broad, I'm not including things like the lip rounding on [ɯ] and the like.
        Transcription occurs in three steps: first, the kana are naively substituted one to one to their correlates.
        For many kana these correlates are indeed their IPA equivalents, but for others it is a kind of intermediate
        representation.
        In the second step, the string is repaired with simple substitutions that reflect how the string is
        actually pronounced.
        In the final step, we conver the remaining non-IPA segments to proper IPA.

        TODO this newest version admits onomatopoeia, which use katakana in really strange ways, and which may represent
        a stratum of Japanese that abides by different phonotactics. Should we exclude them, or at least create two
        versions of this function? For the one that abides by standard phonotactics, we can include pointed
        assert statements, and it's much clearer what's going on.
        I feel like this version was made somewhat messy in trying to let certain onomatopoeia through, and I fear
        that it has possibly made other transcriptions incorrect.
        """
        # exceptions: these particular Japanese words are using は was a particle instead of a kana,
        # so it's pronounced 'wa' instead of 'ha' there
        exceptions = {
            'コンニチハ' : 'koɲit͡ɕiɰa',
            'コンバンハ' : 'kombaɰ̃ɰa',
            'コンバチハ' : 'kombat͡ɕiɰa'
        }
        if self.style == TranscriptionStyle.NARROW:
            exceptions['コンバンハ'] = 'kombaũ͍ɰa'
        # I don't want to deal with detecting these so I will just hardcode them.
        # We're working with a particular dataset, and the Japanese language is unlikely to change soon,
        # so I think it's fine to hardcode them.
        if word_in_katakana in exceptions:
            return exceptions[word_in_katakana]

        # first, we use the above dictionary to convert the kana to our intermediate string:
        intermediate = u''
        for i, kana in enumerate(word_in_katakana):
            if kana not in self.katakana_to_intermediate:
                raise UnsupportedKanaError(kana)
            
            if kana == 'ヽ':
                pre = word_in_katakana[i-1]
                intermediate += self.katakana_to_intermediate[pre]
            elif kana == 'ヾ':
                # I'm too lazy to create a huge dictionary mapping each kana to its equivalent with the tenten,
                # so I will exploit the way katakana are encoded in Unicode.
                # The version of a kana with dakuten has unicode ID one greater than its dakutenless counterpart,
                # eg ord('フ') + 1 == ord('ブ')
                pre = word_in_katakana[i-1]
                unicode_id = ord(pre) # already an int
                voiced_kana = chr(unicode_id + 1)
                intermediate += self.katakana_to_intermediate[voiced_kana]
            # elif kana in {'ヮ', 'ヵ', 'ヶ'}: # these kana aren't supported since they have unclear readings
            #     raise UnsupportedKanaError(kana)
            else:
                intermediate += self.katakana_to_intermediate[kana]

        alveolars = {'n', 't', 'd', 'ɾ', 'z', 'ʑ'}
        labials = {'m', 'p', 'b'}
        velars = {'k', 'ɡ'}
        approxs = {'j', 'h', 'ç', 'ɸ', 's', 'ɕ', 'ɰ'} # not quite the right name for this class, but it's not obvious what the common thread is. Good enough for me.
        fricatives = {'z', 'ʑ', 's', 'ɕ', 'h', 'ç', 'ɸ'}
        h_frics = {'h', 'ç', 'ɸ'}
        vowels = {'a', 'i', 'ɯ', 'e', 'o'}
        nasals = {'n', 'm', 'ŋ', 'I', 'U', 'ɴ'}

        realized = ''
        for i, seg in enumerate(intermediate):
            if seg == 'D':
                if i == 0:
                    realized += 'd'
                else:
                    pre = intermediate[i-1]
                    if pre == 'N':
                        realized += 'd'
                    else:
                        pass
            elif seg == 'Q':
                if i == len(intermediate) - 1:
                    # in mass media the sokuon is put at the end of a word to denote a glottal stop
                    realized += 'ʔ'
                else:
                    post = intermediate[i+1]
                    if post == 'D':
                        realized += 'd' # this is uncommon but occurs in some loans, eg 'キッズ' 'kids'
                    elif post in vowels:
                        realized += 'ʔ'
                    else:
                        realized += post
            elif seg == 'R':
                pre = intermediate[i-1]
                assert pre in vowels or pre == 'N' # exactly one word uses R after N, and it's the onamatopoeia NR
                if pre == 'N':
                    realized += 'ɴ'
                else:
                    realized += pre
            elif seg == 'N':
                # while Yamato/Sino words don't have superheavy syllables, loanwords can,
                # so we must take the preceding segment from realized

                if i == len(intermediate) - 1 or i == 0:
                    # -n is pronounced this way not only at the end of words
                    # but also when it is completely alone (in interjections)
                    realized += 'ɴ'
                else:
                    post = intermediate[i+1]
                    if post in alveolars or post == 'D':
                        realized += 'n'
                    elif post in labials:
                        realized += 'm'
                    elif post in velars:
                        realized += 'ŋ'
                    elif (post in vowels) or (post in approxs):
                        pre = realized[-1]
                        # assert pre in vowels
                        if pre == 'i':
                            realized += 'I' # replace with ĩ later: but we want a one-to-one char correspondence for now
                        else:
                            realized += 'U' # replace with ũ͍/ɰ̃ later
            elif seg == 'J':
                # in Yamato/Sino-Japanese words these can only appear after the -i kana,
                # which suggests deeper phonotactic restrictions than orthographic convention.
                # nevertheless, new loanwords may sometimes put these small jV after other
                # kana to get a desired consonant that doesn't otherwise appear, eg bilabial f or dz
                pre = realized[-1]
                d_pre = realized[-2] # necessarily must exist; these are only deployed after syllables with initials

                # drop the preceding 'i' or other vowel, it's absorbed into the palatalization (or perhaps becomes the palatalization?)
                realized = realized[:-1]
                if (d_pre in fricatives) or (d_pre == 'ɲ'):
                    pass # drop the J, it is neutralized by the segment here
                else:
                    realized += 'ʲ' # palatalize preceding consonant
            elif seg == 'ɯ':
                if i == 0:
                    realized += seg
                else:
                    pre = intermediate[i-1]
                    if pre == 'o':
                        realized += 'o'
                    else:
                        realized += 'ɯ'
            elif seg == 'i':
                if i == 0:
                    realized += seg
                else:
                    pre = intermediate[i-1]
                    if pre == 'e':
                        realized += 'e'
                    else:
                        realized += seg
            else:
                realized += seg

        # we created our finished string without diacritics for ease of string manipulation;
        # now that we're done applying phonological transformations we can polish up the
        # string by making it proper IPA.
        polished = ''
        for i, char in enumerate(realized):
            if char == 't':
                post = realized[i+1]
                if post == 's' or post == 'ɕ':
                    polished += 't͡' # create the affricate with the tie
                else:
                    polished += 't'
            elif char == 'd':
                post = realized[i+1]
                if post == 'z' or post == 'ʑ':
                    polished += 'd͡' # create the affricate
                else:
                    polished += 'd'
            elif char == 'I':
                polished += 'ĩ'
            elif char == 'U':
                if self.style == TranscriptionStyle.NARROW:
                    polished += 'ũ͍' # the nasal vowels is also transcribed [ɰ̃] at times
                elif self.style == TranscriptionStyle.BROAD:
                    polished += 'ɰ̃'
                else:
                    polished += 'ɰ̃'
            elif char in vowels:
                if i == 0:
                    polished += char
                else:
                    pre = realized[i-1]
                    if char == pre and polished[-1] != 'ː':
                        # the second clause is to prevent vowels from being >2 morae in length
                        # in words like 'ＪＡＡ, ジェーエーエー'
                        # which should be transcribed /d͡ʑieːeːeː/, not /d͡ʑieːːːːː/
                        # For a non-acronym example, there is '明王, ミョウオウ,mʲo' 'Wisdom King [of Buddhism]'
                        # which should be /mʲoːoː/, not /mʲoːːː/
                        # TODO I also wonder if this phenomenon further encourages adding syllable separators
                        polished += 'ː'
                    else:
                        # if char == 'ɯ':
                        #     polished += 'ɨ' # not sure how narrow a transcription we want, might comment this out
                        polished += char
            else:
                polished += char

        return polished

    def ipa_to_shorthand(self, ipa):
        '''
        Converts a word in IPA to a shorthand version of the word where one char
        corresponds to one segment (while in IPA, one segment may be a digraph or have diacritics)
        '''
        # we have to be careful about the order in which we make the subs since
        # they can bleed each other, so we should do the substitutions in a particular order
        shorthand = ipa
        for ipa_glyph in self.ipa_sub_order:
            shorthand_seg = self.ipa_to_shorthand_dict[ipa_glyph]
            shorthand = shorthand.replace(ipa_glyph, shorthand_seg)
        return shorthand

    def shorthand_to_fv(self, shorthand):
        '''
        Converts a word in shorthand to a list of its segments' feature vectors
        '''
        result = []
        for c in shorthand:
            result.append(self.shorthand_to_fv_dict[c])
        return result

    def fv_to_binary_fv(self, fvs):
        '''
        Converts a normal list of feature vectors, which has values in {-1, 0, +1},
        to a binary fv with values only in {0, 1}; the -1s are compressed to 0
        '''
        result = []
        for fv in fvs:
            result.append([1 if v == 1 else 0 for v in fv])
        return result

    def fv_to_multihot(self, fvs):
        """
        Converts a list of feature vectors in their numeric forms
        (an entry for each feature; entry equal to +1, -1, 0 for +, -, 0 on that feature, respectively)
        to a list of their multi-hot vector equivalents.
        That is, we convert each feature to a one hot vector, then concatenate the one-hots into a multi-hot vector.

        Strictly speaking this isn't a multilabel vector, as the entries of these vectors
        are not mutually independent. Having a 1 in the [+ant] entry necessarily
        implies having 0's in the [-ant] and [0ant] entries, even if these three entries
        are indeed independent of the entries for, say, [son].
        """
        val_to_onehot = {
            1: [1, 0, 0],
            -1: [0, 1, 0],
            0: [0, 0, 1]
        }
        result = []
        for fv in fvs:
            multihot_fv = []
            for feature_val in fv:
                multihot_fv.extend(val_to_onehot[feature_val])
            result.append(multihot_fv)
        return result

    def katakana_to_romaji(self, word_in_katakana):
        # TODO implement similar to katakana_to_ipa; create a csv with the transcription equivalents (and special chars)
        pass

    def ipa_to_panphon_segments(self, word_in_ipa):
        """
        Converts a string of IPA characters to a list of panphon Segments.
        See the panphon documentation for properties of Segment (https://github.com/dmort27/panphon)
        """
        ft = panphon.FeatureTable()
        return ft.word_fts(word_in_ipa)

    def ipa_to_feature_vectors(self, word_in_ipa):
        """
        Converts a word in IPA to a numpy array of integers {+1,-1,0} corresponding to features.
        A given row corresponds to a segment's list of features. In that list, each index corresponds to a particular feature,
        and the value in {+1,-1,0} corresponds to the feature's value in the natural way.
        We remove the last two features corresponding to tone, so the returned features are, in this order:
        ['syl', 'son', 'cons', 'cont', 'delrel', 'lat', 'nas', 'strid', 'voi', 'sg', 'cg', 'ant', 'cor', 'distr', 'lab', 'hi', 'lo', 'back', 'round', 'velaric', 'tense', 'long']
        See the panphon documentation for specifics of the ordering of the features (https://github.com/dmort27/panphon)
        """
        ft = panphon.FeatureTable()
        # we remove the last two features, which correspond to tone and aren't used in my modeling of Japanese
        return [v[:-2] for v in ft.word_to_vector_list(word_in_ipa, numeric=True)] # TODO investigate role of "normalization"?
