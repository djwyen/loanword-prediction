from typing import List, Mapping

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
        Transcription occurs in two steps: first, the kana are naively substituted one to one to their correlates.
        For many kana these correlates are indeed their IPA equivalents, but for others it is a kind of intermediate
        representation.
        In the second step, the string is repaired with simple regex substitutions.
        
        TODO convert to regex for speed
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
            'ー' : 'ー', # makes the preceding vowel long. Loanwords more often use this, the 'chōonpu', than native words, which simply write the vowel.
        }

        # first, we use the above dictionary to convert the kana to our intermediate string:
        intermediate = u''
        for kana in word:
            assert(kana in katakana_to_intermediate)
            intermediate += katakana_to_intermediate[kana]

        # now we process/repair the intermediate
        # the order of some of these is quite important! these effects are relevant:
        # 1. consonant gemination bleeds affricates leniting into fricatives, eg
        #     eg one gets こっち realized as 'kotchi', not 'kosshi'
        # 2. we must decide whether to realize D before realizing geminates, placement of N, etc.
        #     because we want to naively be able to take the next segment and examine its features
        #     (the same is not true of J since it necessarily is not preceded by a sokuon/-n)
        # TODO I think the below can be done in one pass? we can more or less preserve the invariant that anything appended to intermediate is finished. The only thing that requires a lookback is the small -jV kana but they never modify the preceding consonant. Geminate consonants and -n require a lookahead, which can interfere with unrealized 'D', but you can handle that explicitly.

        repaired = ''

        # using these for realization of -n and palatal assimilation
        alveolars = {'n', 't', 'd', 'ɾ', 'z', 'ʑ'}
        labials = {'m', 'p', 'b'}
        velars = {'k', 'ɡ'}
        approxs = {'j', 'h', 'ç', 'ɸ', 's', 'ɕ', 'ɰ'} # not quite the right name for this class, but it's not obvious what the common thread is. Good enough for me.
        fricatives = {'z', 'ʑ', 's', 'ɕ', 'h', 'ç', 'ɸ'}
        h_frics = {'h', 'ç', 'ɸ'}
        vowels = {'a', 'i', 'ɯ', 'e', 'o'}
        nasals = {'n', 'm', 'ŋ', 'ĩ', 'ũ͍', 'ɴ'}


        for i, seg in enumerate(intermediate):
            if seg == 'D':
                if i == 0:
                    repaired += 'd'
                else:
                    pre = intermediate[i-1]
                    if pre == 'N':
                        repaired += 'd'
                    else:
                        pass # d disappears intervocalically or after geminates, which this must be.
            else:
                repaired += seg
        
        intermediate = repaired
        repaired = ''

        # substitute Q with an actual geminate segment; realize N; realize long vowels; realize palatals
        # we can do the first three simultaneously since they necessarily do not interfere with each other:
        # Japanese has no superheavy syllables.
        # as for the last, it can be done simultaneously since it only depends on the segments before it
        # honestly though the performance speedup is so minimal and I worry that I am missing some bizarre edge case,
        # especially since I don't actually speak Japanese...
        for i, seg in enumerate(intermediate):
            if seg == 'Q':
                post = intermediate[i+1] # necesarily must exist; this is deployed before another syllable
                repaired += post
            elif seg == 'N':
                # unless we're transcribing Bantu loanwords -n is necessarily preceded by a vowel,
                # but one can never be too careful
                pre = intermediate[i-1]
                assert pre in vowels
                if i == len(intermediate) - 1:
                    repaired += 'ɴ'
                else:
                    post = intermediate[i+1]
                    if post in alveolars:
                        repaired += 'n'
                    elif post in labials:
                        repaired += 'm'
                    elif post in velars:
                        repaired += 'ŋ'
                    elif (post in vowels) or (post in approxs):
                        if pre == 'i':
                            repaired += 'ĩ' # TODO I think this counts as two chars, make sure this doesn't screw things up
                        else:
                            repaired += 'ũ͍'
            elif seg == 'ー':
                pre = intermediate[i-1] # necessarily must exist; never deployed after a closed syllable
                assert pre in vowels
                repaired += pre
            elif seg == 'J':
                # note that due to orthographic convention these can only appear after the -i kana,
                # which is very useful to us (and suggests it may be deeper than just convention)
                # consider the h- series: the palatalized versions always have the cedilla c
                pre = intermediate[i-1]
                assert pre == 'i'
                d_pre = intermediate[i-2] # necessarily must exist; these are only deployed after syllables with initials

                # drop the preceding 'i', it's absorbed into the palatalization (or perhaps becomes the palatalization?)
                repaired = repaired[:-1]
                if (d_pre in fricatives) or (d_pre == 'ɲ'):
                    pass # drop the J, it is neutralized by the segment here
                else:
                    repaired += 'ʲ' # palatalize preceding consonant
            else:
                repaired += seg

        return repaired

    def convert_kana(self, kana: str) -> List[int]:
        '''Converts all kana but -n, since that undergoes place assimilation'''

        # idea: create a dict like the kana_to_ipa dict above that simply transcribes each kana naively, then goes through and repairs everything to look like real Japanese.
        # I mean, you were already going to do something like this given the existence of Japanese -n and the "geminating" kana
        # this shouldn't be too bad, the only real rules you have to handle are the palatalization small kana, nasal assimilation, and investigating whether dzi really is intervocalically modulated or not
        return kana_to_ipa[kana]

    def convert_word(self, word: str) -> List[int]:
        pass

if __name__ == '__main__':
    t = Transcriber()
    t.katakana_to_ipa()