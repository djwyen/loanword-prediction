from typing import List, Mapping
from data_processing.data_errors import UnsupportedKanaError

# TODO probably we don't need a transcriber class? but it is nice to have the names here isolated, so consider it.

class Transcriber():
    '''
    A class to wrap dictionaries for converting Japanese words between scripts:
    - Kana to IPA
    - Kana to Romaji
    - IPA to feature vectors
    Both hiragana and katakana are supported
    '''

    katakana = {'ァ', 'ア', 'ィ', 'イ', 'ゥ', 'ウ', 'ェ', 'エ', 'ォ', 'オ', 'カ', 'ガ', 'キ', 'ギ', 'ク', 'グ', 'ケ', 'ゲ', 'コ', 'ゴ', 'サ', 'ザ', 'シ', 'ジ', 'ス', 'ズ', 'セ', 'ゼ', 'ソ', 'ゾ', 'タ', 'ダ', 'チ', 'ヂ', 'ッ', 'ツ', 'ヅ', 'テ', 'デ', 'ト', 'ド', 'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 'ハ', 'バ', 'パ', 'ヒ', 'ビ', 'ピ', 'フ', 'ブ', 'プ', 'ヘ', 'ベ', 'ペ', 'ホ', 'ボ', 'ポ', 'マ', 'ミ', 'ム', 'メ', 'モ', 'ャ', 'ヤ', 'ュ', 'ユ', 'ョ', 'ヨ', 'ラ', 'リ', 'ル', 'レ', 'ロ', 'ヮ', 'ワ', 'ヰ', 'ヱ', 'ヲ', 'ン', 'ヴ', 'ヵ', 'ヶ', 'ヽ', 'ヾ'}

    # TODO question: correct way to represent long vowels on the IPA front? it's attractive to just literally write out say /kaa/ for 'カァ' since it involves no lookahead, unlike /ka:/. Perhaps this will ultimately depend on how we represent length in the feature vector: whether it gets its own dimension.
    # kana_to_ipa = {
    #     'ァ' : 'a', 'ア' : 'a',
    #     'ィ' : 'i', 'イ' : 'i',
    #     'ゥ' : 'ɯ', 'ウ' : 'ɯ', 
    #     'ェ' : 'e', 'エ' : 'e', 
    #     'ォ' : 'o', 'オ' : 'o', 
        
    #     'カ' : 'ka', 'ガ' : 'ɡa', # be careful! IPA 'ɡ' is NOT the same as the g you type on your keyboard
    #     'キ' : 'ki', 'ギ' : 'ɡi',
    #     'ク' : 'kɯ', 'グ' : 'ɡɯ',
    #     'ケ' : 'ke', 'ゲ' : 'ɡe',
    #     'コ' : 'ko', 'ゴ' : 'ɡo',    
        
    #     'サ' : 'sa', 'ザ' : 'za',
    #     'シ' : 'ɕi', 'ジ' : '(d)ʑi', # address this, it seems the d is dropped intervocalically (but retained otherwise, ie at beginning of words or after a syllable closed by -n)
    #     'ス' : 'sɯ', 'ズ' : '(d)zɯ', # same as with dzi above
    #     'セ' : 'se', 'ゼ' : 'ze',
    #     'ソ' : 'so', 'ゾ' : 'zo',
          
    #     'タ' : 'ta', 'ダ' : 'da',
    #     'チ' : 't͡ɕi', 'ヂ' : '(d)ʑi', # this is rarely used, consider dropping it
    #     'ッ' : 'Q', # this precedes a geminate consonant
    #     'ツ' : 't͡sɯ', 'ヅ' : '(d)zɯ', # as above, this is rarely used
    #     'テ' : 'te', 'デ' : 'de',
    #     'ト' : 'to', 'ド' : 'do',
        
    #     'ナ' : 'na', 'ニ' : 'ɲi', 'ヌ' : 'nɯ', 'ネ' : 'ne', 'ノ' : 'no',
        
    #     'ハ' : 'ha', 'バ' : 'ba', 'パ' : 'pa',
    #     'ヒ' : 'çi', 'ビ' : 'bi', 'ピ' : 'pi',
    #     'フ' : 'ɸɯ', 'ブ' : 'bɯ', 'プ' : 'pɯ',
    #     'ヘ' : 'he', 'ベ' : 'be', 'ペ' : 'pe',
    #     'ホ' : 'ho', 'ボ' : 'bo', 'ポ' : 'po',

    #     'マ' : 'ma', 'ミ' : 'mi', 'ム' : 'mɯ', 'メ' : 'me', 'モ' : 'mo',
    #     'ャ' : 'ʲa', 'ュ' : 'ʲɯ', 'ョ' : 'ʲo', # these combine to palatalize preceding kana, but there are some rules that apply. The palatalization is neutralized when the initial consonant is a fricative or affricate. Consider encoding this in a different way?
    #     'ヤ' : 'ja', 'ユ' : 'jɯ', 'ヨ' : 'jo',
        
    #     'ラ' : 'ɾa', 'リ' : 'ɾi', 'ル' : 'ɾɯ', 'レ' : 'ɾe', 'ロ' : 'ɾo',
    #     'ヮ' : '', # this is obsolete, Japanese used to have rounded syllables but those have lost their rounding
    #     'ワ' : 'ɰa', 
    #     'ヰ' : 'i', # historically [wi] but this has changed, and the kana is largely obsolete
    #     'ヱ' : 'e', # same as with ヰ, this used to be [we]
    #     'ヲ' : 'o', # like the above but less obsolete, sometimes used as a particle
    #     'ン' : 'ɴ', # encode the place assimilation to [n m ŋ] somehow? and the changes between [ɴ ɰ̃]
    #     'ヴ' : 'bɯ', # sometimes [vu], only in loanwords
    #     'ヵ' : 'ka', # only used as a counter word
    #     'ヶ' : '', # can be ka ga and ko depending on context
    #     'ヽ' : '', # iteration mark, equivalent to 々. Repeats preceding syllable
    #     'ヾ' : '', # repeats preceding syllable and voices it
    #     'ー' : 'ː', # makes the preceding vowel long. Loanwords more often use this, the 'chōonpu', than native words, which simply write the vowel.
    # }

    # NOTE to self: the above copy paste took a ridiculously long time and revealed to me that I should probably investigate another way of sourcing pronunciations of Japanese words rather than rely on the kana entries: there are some relatively complicated rules changing pronunciation as kana combine. They aren't so bad that you can't approximate them well with a one-char lookahead substitution, but given that there are some kana that can vary in their reading, one should consider finding other ways of transcribing.

    def katakana_to_ipa(self, word: str) -> str:
        '''
        Converts a word or series of words in katakana to its IPA equivalent.
        
        The transcription is pretty broad, I'm not including things like the lip rounding on [ɯ] and the like.
        Transcription occurs in three steps: first, the kana are naively substituted one to one to their correlates.
        For many kana these correlates are indeed their IPA equivalents, but for others it is a kind of intermediate
        representation.
        In the second step, the string is repaired with simple substitutions that reflect how the string is
        actually pronounced.
        In the final step, we conver the remaining non-IPA segments to proper IPA.

        TODO convert to regex for speed
        TODO this newest version admits onomatopoeia, which use katakana in really strange ways, and which may represent
        a stratum of Japanese that abides by different phonotactics. Should we exclude them, or at least create two
        versions of this function? For the one that abides by standard phonotactics, we can include pointed
        assert statements, and it's much clearer what's going on.
        I feel like this version was made somewhat messy in trying to let certain onomatopoeia through, and I fear
        that it has possibly made other transcriptions incorrect.
        '''

        katakana_to_intermediate = {
            'ァ' : 'a', 'ア' : 'a',
            'ィ' : 'i', 'イ' : 'i',
            'ゥ' : 'ɯ', 'ウ' : 'ɯ', 
            'ェ' : 'e', 'エ' : 'e', 
            'ォ' : 'o', 'オ' : 'o', 
            
            'カ' : 'ka', 'ガ' : 'ɡa', # be careful! IPA 'ɡ' is NOT the same as the g you type on your keyboard
            'キ' : 'ki', 'ギ' : 'ɡi',
            'ク' : 'kɯ', 'グ' : 'ɡɯ',
            'ケ' : 'ke', 'ゲ' : 'ɡe',
            'コ' : 'ko', 'ゴ' : 'ɡo',

            'サ' : 'sa', 'ザ' : 'za',
            'シ' : 'ɕi', 'ジ' : 'Dʑi', # making the executive decision to transcribe the initial consonant of the
            'ス' : 'sɯ', 'ズ' : 'Dzɯ', # yotsugana as a fricative intervocalically and an affricate
            'セ' : 'se', 'ゼ' : 'ze', # otherwise (beginning of words, or after closed syllables)
            'ソ' : 'so', 'ゾ' : 'zo', # I used wiktionary on various words like 杏子, 小豆, to help support this interpretation.
            # also see https://en.wikipedia.org/wiki/Hiragana which claims "In many accents, the j and z sounds are pronounced as affricates at the beginning of utterances and fricatives in the middle of words". Post-n is the only ambiguous case but wiktionary "anzu" above suggests it is also an affricate there.
            # anyway, more relevantly, I use capital D for a possibly deleted d.

            'タ' : 'ta', 'ダ' : 'da',
            'チ' : 'tɕi', 'ヂ' : 'Dʑi', # encoding 'ts' without the tie since it messes up things; will add it to
            'ツ' : 'tsɯ', 'ヅ' : 'Dzɯ', # ts/dz later
            'テ' : 'te', 'デ' : 'de',
            'ト' : 'to', 'ド' : 'do',
            'ッ' : 'Q', # this precedes a geminate consonant

            'ナ' : 'na', 'ニ' : 'ɲi', 'ヌ' : 'nɯ', 'ネ' : 'ne', 'ノ' : 'no',

            'ハ' : 'ha', 'バ' : 'ba', 'パ' : 'pa',
            'ヒ' : 'çi', 'ビ' : 'bi', 'ピ' : 'pi',
            'フ' : 'ɸɯ', 'ブ' : 'bɯ', 'プ' : 'pɯ',
            'ヘ' : 'he', 'ベ' : 'be', 'ペ' : 'pe',
            'ホ' : 'ho', 'ボ' : 'bo', 'ポ' : 'po',

            'マ' : 'ma', 'ミ' : 'mi', 'ム' : 'mɯ', 'メ' : 'me', 'モ' : 'mo',
            'ャ' : 'Ja', 'ュ' : 'Jɯ', 'ョ' : 'Jo', # using capital J for a glide that is possibly either deleted or assimilated to the preceding consonant
            'ヤ' : 'ja', 'ユ' : 'jɯ', 'ヨ' : 'jo',

            'ラ' : 'ɾa', 'リ' : 'ɾi', 'ル' : 'ɾɯ', 'レ' : 'ɾe', 'ロ' : 'ɾo',
            'ヮ' : '', # this is obsolete, don't think it will be encountered.
            'ワ' : 'ɰa',
            'ヰ' : 'i',
            'ヱ' : 'e',
            'ヲ' : 'o',
            'ン' : 'N', # encoding nasal as N; place assimilation and the like to come later
            'ヴ' : 'bɯ',
            'ヵ' : '', # only used as a counter word, not encoding it
            'ヶ' : '', # can be ka ga and ko depending on context; not encoding it
            'ヽ' : 'ヽ', # iteration mark, equivalent to 々. Repeats preceding syllable
            'ヾ' : 'ヾ', # repeats preceding syllable and voices it
            'ー' : 'R', # encoding long vowel as R
        }
        
        # exceptions: these particular Japanese words are using は was a particle instead of a kana,
        # so it's pronounced 'wa' instead of 'ha' there
        exceptions = {
            'コンニチハ' : 'koɲit͡ɕiɰa',
            'コンバンハ' : 'kombaũ͍ɰa',
            'コンバチハ' : 'kombat͡ɕiɰa'
        }
        # I don't want to deal with detecting these so I will just hardcode them.
        # We're working with a particular dataset, and the Japanese language is unlikely to change soon,
        # so I think it's fine to hardcode them.
        if word in exceptions:
            return exceptions[word]

        # first, we use the above dictionary to convert the kana to our intermediate string:
        intermediate = u''
        for i, kana in enumerate(word):
            try:
                assert(kana in katakana_to_intermediate)
            except:
                return None # a way to signal that this isn't a transcribable word
                # print('kana not found:', kana)
            
            if kana == 'ヽ':
                pre = word[i-1]
                intermediate += katakana_to_intermediate[pre]
            elif kana == 'ヾ':
                # I'm too lazy to create a huge dictionary mapping each kana to its equivalent with the tenten,
                # so I will exploit the way katakana are encoded in Unicode.
                # The version of a kana with dakuten has unicode ID one greater than its dakutenless counterpart,
                # eg ord('フ') + 1 == ord('ブ')
                pre = word[i-1]
                unicode_id = ord(pre) # already an int
                voiced_kana = chr(unicode_id + 1)
                intermediate += katakana_to_intermediate[voiced_kana]
            elif kana in {'ヮ', 'ヵ', 'ヶ'}: # these kana aren't supported since they have unclear readings
                raise UnsupportedKanaError(kana)
            else:
                intermediate += katakana_to_intermediate[kana]

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
                            realized += 'U' # replace with ũ͍ later
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
                polished += 'ũ͍'
            elif char == 'ɯ':
                polished += 'ɨ' # not sure how narrow a transcription we want, might comment this out
            else:
                polished += char

        return polished

    def convert_kana(self, kana: str) -> List[int]:
        '''Converts all kana but -n, since that undergoes place assimilation'''

        # idea: create a dict like the kana_to_ipa dict above that simply transcribes each kana naively, then goes through and repairs everything to look like real Japanese.
        # I mean, you were already going to do something like this given the existence of Japanese -n and the "geminating" kana
        # this shouldn't be too bad, the only real rules you have to handle are the palatalization small kana, nasal assimilation, and investigating whether dzi really is intervocalically modulated or not
        return kana_to_ipa[kana]

    def convert_word(self, word: str) -> List[int]:
        pass

# if __name__ == '__main__':
#     t = Transcriber()
#     t.katakana_to_ipa()
