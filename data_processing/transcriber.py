import csv
from typing import List, Mapping
from data_processing.data_errors import UnsupportedKanaError

import numpy as np
import panphon
from panphon.segment import Segment

KATAKANA_TO_IPA_CSV = 'kana_to_ipa.csv'

# represents how strict to be with transcriptions, currently just two settings
class TranscriptionStyle:
    BROAD = 1
    NARROW = 2

TRANSCRIPTION_STYLE = TranscriptionStyle.BROAD # can be set via a commandline argument, eventually

class Transcriber():
    """
    A class to wrap dictionaries for converting Japanese words between scripts:
    - Kana to IPA
    - Kana to Romaji
    - IPA to feature vectors
    Both hiragana and katakana are supported
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

    def katakana_to_ipa(self, word: str) -> str:
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
        if word in exceptions:
            return exceptions[word]

        # first, we use the above dictionary to convert the kana to our intermediate string:
        intermediate = u''
        for i, kana in enumerate(word):
            if kana not in self.katakana_to_intermediate:
                raise UnsupportedKanaError(kana)
            
            if kana == 'ヽ':
                pre = word[i-1]
                intermediate += self.katakana_to_intermediate[pre]
            elif kana == 'ヾ':
                # I'm too lazy to create a huge dictionary mapping each kana to its equivalent with the tenten,
                # so I will exploit the way katakana are encoded in Unicode.
                # The version of a kana with dakuten has unicode ID one greater than its dakutenless counterpart,
                # eg ord('フ') + 1 == ord('ブ')
                pre = word[i-1]
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
        approxs = {'j', 'h', 'ç', 'ɸ', 's', 'ɕ', 'ɰ'} # not quite the right name for this class, but it's not obvious what the common thread is. Good enough for me.
        fricatives = {'z', 'ʑ', 's', 'ɕ', 'h', 'ç', 'ɸ'}
        h_frics = {'h', 'ç', 'ɸ'}
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
                            realized += 'I' # replace with ĩ later: but we want a one-to-one char correspondence for now
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
                polished += 'ĩ'
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
                    if char == pre:
                        polished += 'ː'
                    else:
                        # if char == 'ɯ':
                        #     polished += 'ɨ' # not sure how narrow a transcription we want, might comment this out
                        polished += char
            else:
                polished += char

        return polished

    def ipa_to_panphon_word(self, word: str) -> Segment:
        """
        Converts a string of IPA characters to their feature vectors.
        """
        ft = panphon.FeatureTable()
        assert(ft.validate_word(word))
        return ft.word_fts(word)

    def ipa_to_numpy_array(self, word: str) -> List[List[int]]:
        """
        Converts a word in IPA to a numpy feature array
        """
        ft = panphon.FeatureTable()
        assert(ft.validate_word(word))
        return ft.word_to_vector_list(word, numeric=True) # TODO investigate role of "normalization"?
