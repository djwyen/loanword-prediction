This document compiles some comments on the kana transcription choices I made that were present in `data_processing/transcriber.py` back when the kana conversion was stored in a dictionary, but had to be removed when the choice was made to (responsibly) move that table to a .csv instead.

- Be careful! IPA 'ɡ' is NOT the same as the g you type on your keyboard

- Apr 27, 2022: I encountered another bug on this pattern, and it's worse since the characters render identically. IPA 'ç' (or at least, the one panphon recognizes) is distinct from non-IPA 'ç'. In particular the former is actually a digraph in unicode. I will consider opening tickets on Panphon about this and 'g' above...

- Apr 27, 2022: The same bug but with IPA 'ĩ' which is distinct from 'ĩ'. I've run into several problems of this sort copy-pasting from wikipedia...

- I make the executive decision to transcribe the initial consonant of the yotsugana as a fricative intervocalically and an affricate otherwise (beginning of words or after closed syllables). I used wiktionary on various words like 杏子, 小豆 to help support this interpretation. Also see https://en.wikipedia.org/wiki/Hiragana which claims "In many accents, the j and z sounds are pronounced as affricates at the beginning of utterances and fricatives in the middle of words". Post-n is the only ambiguous case but wiktionary "anzu" above suggests it is also an affricate there.

- I encode /ts/ without the tie since it messes up things; will add it to ts/dz later

- capital 'D' denotes for a possibly deleted /d/.

- capital 'J' denotes a glide that is possibly either deleted or assimilated to the preceding consonant

- capital 'Q' is used to denote the segment preceding a geminate consonant

- capital 'N' denotes the word final nasal; this allows us to later substitute it to mark place assimilation and the like

- capital 'R' denotes a long vowel.

- the kana 'ヮ' is obsolete and not encoded; it is unlikely to be encountered

- the kana 'ヵ' is only used as a counter word and not encoded

- the kana 'ヶ' can be /ka/, /ga/, and /go/ depending on context. It is rare and not encoded.

- the kana 'ヽ' is the iteration mark, equivalent to '々'. Repeats preceding syllable.

- the kana 'ヾ' repeats the preceding syllable but voices it.
