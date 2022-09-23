# loanword-prediction
Machine learning project for predicting loanwords

When speakers of a given language borrow words from another language, they will modify those words to conform to the phonology of their native language. They will substitute sounds that do not exist in their language with the ones they deem the closest, modify syllables to conform to their language's syllable structure, et cetera. For example, consider these borrowings into Japanese:

| Loanword | Romanization | Original Word |
| -------- | ------------ | ------------- |
| アイスクリーム | aisu kurīmu | English _ice cream_ |
| アパート | apāto | English _apartment_ |
| アルバイト | arubaito | German _Arbeit_ "work" |
| ラッコ | rakko | Ainu _rakko_ "sea otter" |
| トマト | tomato | Nahuatl _tomato_ "tomato" |

Let us examine the first borrowing, "ice cream" -> "aisu kurīmu". A Japanese syllable must be of the form (C)(j)V(Q/N), where C is a consonant, Q is a geminated consonant, and N is a generic nasal. The English word "ice" /aɪs/ does not conform to this syllable structure due to the presence of a coda /-s/, so it is repaired to /aisu/ through the epenthesis (insertion) of a /u/. Note that this causes the segment /s/ to change from a final consonant to an initial consonant.

Similarly, the English word "cream" /kɹiːm/ contains both a coda /-m/ as before, which is also repaired with epenthesis of a /u/. Additionally, it contains a consonant cluster /kɹ/, which is invalid in Japanese. The cluster is also repaired by epenthesis of a /u/. Several questions immediately arise: why is epenthesis of a /u/ the most common repair strategy? Why could it not have been a different vowel, such as /a/ or /i/? And why is epenthesis the strategy at all, when one could have instead deleted offending consonants and produced a form like */aɪ kīmu/? Why is the final m in "cream" repaired when syllable-final m is tolerated in Japanese? (This last question has a more ready answer: m has a place specification while the Japanese nasal generally lacks a place feature. But the question remains in a more abstract form: why not repair the segment by eliminating the place feature, yielding a form like */aisu kurīn/?)

TODO finish copy


## Tasks:
- [X] Identify source of compute for this project, eg Google Colab or a professor's servers (used Google Colab)
- [X] Identify which language to study in this project (will be doing Japanese — its phonotactics and loanword acquisition are well-studied. Furthermore, Japanese has rather simple phonotactics, so it seems like an easier task for a model to learn; and it has copious amounts of well-documented recent loanwords.)
- [X] Find a corpus representing naturalistic speech in this language (BCCWJ seems good)
- [ ] Find lists of loanwords with etymologies for this language, perhaps by scraping Wiktionary
- [X] Design an autoencoder model to learn this task. RNN seems well-suited, but a CNN may be useful for long distance effects that could be relevant: vowel harmony in Turkish, Lyman's law in Japanese. A transformer may be well-suited but seems overkill. (used RNN)
- [X] Summarize results in a writeup

## Datasets:

### Naturalistic Japanese (including Yamato, Sino-Japanese, and loanwords)

- The Balanced Corpus of Contemporary Written Japanese (BCCWJ) [hosted here](https://ccd.ninjal.ac.jp/bccwj/en/index.html).
    - The word frequency list (`BCCWJ_frequencylist_suw_ver1_0.zip`) and its manual (`BCCWJ_frequencylist_manual_ver1_0.pdf`) are [available here](https://ccd.ninjal.ac.jp/bccwj/en/freq-list.html)

### Japanese loanwords and their etymologies
- https://japanesetactics.com/english-gairaigo-list-learn-301-japanese-words-in-10-minutes
- idea: there may be some unconventional sources of data, such as transcriptions of demon names in Shin Megami Tensei. In general video games and anime may provide a source of nativizations, and possibly reflect more recent trends that older loans like the above will not have. On the other hand though, it's not clear that romanizations of say names will use the exact same phonological process as the above, which were not as intentional.
- scraping Japanese Wiktionary, such as [lists of Gairaigo](https://ja.wiktionary.org/w/index.php?title=%E3%82%AB%E3%83%86%E3%82%B4%E3%83%AA:%E6%97%A5%E6%9C%AC%E8%AA%9E_%E8%8B%B1%E8%AA%9E%E7%94%B1%E6%9D%A5&pageuntil=%E3%81%88%E3%81%8F%E3%81%99%E3%81%BB%E3%81%8A%E3%81%A8+%E3%81%88%E3%81%8F%E3%81%99%E3%81%BD%E3%81%8A%E3%81%A8%0A%E3%82%A8%E3%82%AF%E3%82%B9%E3%83%9D%E3%83%BC%E3%83%88#mw-pages)?

## Readings:
- [Morita, Takashi (2018). Unsupervised learning of lexical subclasses from phonotactics](https://dspace.mit.edu/handle/1721.1/120612)

## Potentially useful tools:
- https://github.com/bootphon/phonemizer to convert Japanese to IPA
- https://github.com/dmort27/panphon to convert IPA to feature vectors
- https://github.com/dmort27/epitran as another way to convert IPA to feature vectors
- https://lingpy.org/ to align words
