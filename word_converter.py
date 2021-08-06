from typing import List, Mapping


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
    kana_to_ipa = {
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
        'シ' : 'ɕi', 'ジ' : '(d)ʑi', # address this, it seems the d is dropped intervocalically (but retained otherwise, ie at beginning of words or after a syllable closed by -n)
        'ス' : 'sɯ', 'ズ' : '(d)zɯ', # same as with dzi above
        'セ' : 'se', 'ゼ' : 'ze',
        'ソ' : 'so', 'ゾ' : 'zo',
          
        'タ' : 'ta', 'ダ' : 'da',
        'チ' : 't͡ɕi', 'ヂ' : '(d)ʑi', # this is rarely used, consider dropping it
        'ッ' : 'Q', # this precedes a geminate consonant
        'ツ' : 't͡sɯ', 'ヅ' : '(d)zɯ', # as above, this is rarely used
        'テ' : 'te', 'デ' : 'de',
        'ト' : 'to', 'ド' : 'do',
        
        'ナ' : 'na', 'ニ' : 'ɲi', 'ヌ' : 'nɯ', 'ネ' : 'ne', 'ノ' : 'no',
        
        'ハ' : 'ha', 'バ' : 'ba', 'パ' : 'pa',
        'ヒ' : 'çi', 'ビ' : 'bi', 'ピ' : 'pi',
        'フ' : 'ɸɯ', 'ブ' : 'bɯ', 'プ' : 'pɯ',
        'ヘ' : 'he', 'ベ' : 'be', 'ペ' : 'pe',
        'ホ' : 'ho', 'ボ' : 'bo', 'ポ' : 'po',

        'マ' : 'ma', 'ミ' : 'mi', 'ム' : 'mɯ', 'メ' : 'me', 'モ' : 'mo',
        'ャ' : 'ʲa', 'ュ' : 'ʲɯ', 'ョ' : 'ʲo', # these combine to palatalize preceding kana, but there are some rules that apply. The palatalization is neutralized when the initial consonant is a fricative or affricate. Consider encoding this in a different way?
        'ヤ' : 'ja', 'ユ' : 'jɯ', 'ヨ' : 'jo',
        
        'ラ' : 'ɾa', 'リ' : 'ɾi', 'ル' : 'ɾɯ', 'レ' : 'ɾe', 'ロ' : 'ɾo',
        'ヮ' : '', # this is obsolete, Japanese used to have rounded syllables but those have lost their rounding
        'ワ' : 'ɰa', 
        'ヰ' : 'i', # historically [wi] but this has changed, and the kana is largely obsolete
        'ヱ' : 'e', # same as with ヰ, this used to be [we]
        'ヲ' : 'o', # like the above but less obsolete, sometimes used as a particle
        'ン' : 'ɴ', # encode the place assimilation to [n m ŋ] somehow? and the changes between [ɴ ɰ̃]
        'ヴ' : 'bɯ', # sometimes [vu], only in loanwords
        'ヵ' : 'ka', # only used as a counter word
        'ヶ' : '', # can be ka ga and ko depending on context
        'ヽ' : '', # iteration mark, equivalent to 々. Repeats preceding syllable
        'ヾ' : '', # repeats preceding syllable and voices it
        'ー' : 'ː', # makes the preceding vowel long. Loanwords more often use this, the 'chōonpu', than native words, which simply write the vowel.
    }

    # NOTE to self: the above copy paste took a ridiculously long time and revealed to me that I should probably investigate another way of sourcing pronunciations of Japanese words rather than rely on the kana entries: there are some relatively complicated rules changing pronunciation as kana combine. They aren't so bad that you can't approximate them well with a one-char lookahead substitution, but given that there are some kana that can vary in their reading, one should consider finding other ways of transcribing.

    def convert_kana(self, kana: str) -> List[int]:
        '''Converts all kana but -n, since that undergoes place assimilation'''

        # idea: create a dict like the kana_to_ipa dict above that simply transcribes each kana naively, then goes through and repairs everything to look like real Japanese.
        # I mean, you were already going to do something like this given the existence of Japanese -n and the "geminating" kana
        # this shouldn't be too bad, the only real rules you have to handle are the palatalization small kana, nasal assimilation, and investigating whether dzi really is intervocalically modulated or not
        return kana_to_ipa[kana]

    def convert_word(self, word: str) -> List[int]:
        pass
