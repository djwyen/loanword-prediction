"""
Quick script to process the data from the Balanced Corpus of Contemporary Written Japanese (BCCWJ), which is linked under the README.

Our dataset is the BCCWJ "Short Unit Word list data" (`BCCWJ_frequencylist_suw_ver1_0.tsv`).
It contains more properties of the words than are necessary for our purposes, such as frequency data,
and also contains tokens that are not useful for us, like symbols and proper nouns.
So this script filters out just the words we need for our project - words constituting a representative part of the Japanese vocabulary
(for training an autoencoder on Japanese phonology) - and also only records the fields we need for each word,
namely its katakana transcription, its lemma, its part of speech, and its sublexical origin (eg Yamato, Sinitic).

This website has some information on labels but it seems less relevant than the manual + Morita (2018)'s comments https://ccd.ninjal.ac.jp/bccwj/en/tsv.html
"""

import os
import csv
from data_processing import transcriber

COUNT_ETYMOLOGIES = False # TODO eventually I will make this a flag for the script

# path to the short BCCWJ word frequency list
PATH_TO_UNPROCESSED_TSV = "data/BCCWJ/BCCWJ_frequencylist_suw_ver1_0.tsv"

# path to the new, processed wordlist I am building
PATH_TO_OUTPUT_CSV = "data/BCCWJ/pared_BCCWJ.csv"

# we are interested in these particular properties:
#   lForm (index 1): a katakana transcription of the given word
#   lemma (index 2): the Japanese written form of the word, in kanji/hiragana/katakana
#   pos   (index 3): the part of speech of the word
#   wType (index 5): the sublexicon of origin for a given word, namely native Yamato (和) Sino-Japanese (漢) or foreign (外).
#       There are also mixed etymology words (混), proper names (固), and symbols (記号),
#       which covers anything not written in a Japanese script (eg acronyms like 'DVD')
#   core_frequency (index 78): following Morita (2018), we only consider words with positive core_frequency, 
#       as it is only those words which were manually checked by humans for validity
YAMATO = '和'
SINO = '漢'
FOREIGN = '外'
MIXED = '混'
PROPER_NAME = '固'
SYMBOLS = '記号'

# the relevant parts-of-speech for certain kinds of words one may want to exclude
PARTICLE = '助詞'
ONOMATOPOEIA = '感動詞'

def main():
    with open(PATH_TO_UNPROCESSED_TSV) as f:
        reader = csv.reader(f, delimiter='\t')

        header = next(reader)
        # `prop_to_index` maps the name of a given property to its index in a data row for ease of use
        # the meaning of the properties is expounded upon in the BCCWJ manual, available at the same place as the actual data.
        # I can't actually read Japanese, but Morita's explanation of the corpus is sufficient for my purposes,
        #   and Google Translate + knowledge of Chinese characters has served as a good aid in filling in the gaps
        prop_to_index = {header[i] : i for i in range(len(header))}

        if COUNT_ETYMOLOGIES:
            yamato_count = 0
            sino_count = 0
            foreign_count = 0
            for entry in reader:
                core_freq = entry[prop_to_index['core_frequency']]
                if core_freq != '' and int(core_freq) > 0:
                    origin_lang = entry[prop_to_index['wType']]
                    if origin_lang == YAMATO:
                        yamato_count += 1
                    elif origin_lang == SINO:
                        sino_count += 1
                    elif origin_lang == FOREIGN:
                        foreign_count += 1
            
            print(f'{yamato_count} Yamato words')
            print(f'{sino_count} Sino-Japanese words')
            print(f'{foreign_count} Foreign words')
            # after applying the core frequency > 0 requirement we reproduce Takashi's counts (at least for SJ and foreign, which he discloses):
            # 9893 Yamato, 13373 Sino-Japanese, 4421 foreign

        with open(PATH_TO_OUTPUT_CSV, 'w+') as f2:
            t = transcriber.Transcriber()
            # to count the etymological distribution of the recorded words
            yamato_count = 0
            sino_count = 0
            foreign_count = 0
            total_count = 0
            
            writer = csv.writer(f2)
            writer.writerow(['word', 'kana', 'ipa', 'origin'])

            for entry in reader:
                core_freq = entry[prop_to_index['core_frequency']]
                pos = entry[prop_to_index['pos']]
                kana = entry[prop_to_index['lForm']]
                word = entry[prop_to_index['lemma']]
                origin_lang = entry[prop_to_index['wType']]

                if core_freq != '' and int(core_freq) > 0:
                    if PARTICLE in pos or ONOMATOPOEIA in pos:
                        pass
                    else:
                        try:
                            ipa = t.katakana_to_ipa(kana)
                            if ipa != "":
                                writer.writerow([word, kana, ipa, origin_lang])

                                total_count += 1
                                if origin_lang == YAMATO:
                                    yamato_count += 1
                                elif origin_lang == SINO:
                                    sino_count += 1
                                elif origin_lang == FOREIGN:
                                    foreign_count += 1
                        except:
                            # the above could fail if an unrecognized kana is used
                            # print(f'Failed to convert to IPA on word {word}')
                            pass
            print(f'Recorded {total_count} words:')
            print(f'- {yamato_count} Yamato words')
            print(f'- {sino_count} Sino-Japanese words')
            print(f'- {foreign_count} foreign words')
            # when this script was last run we produced 36396 words: 9667 Yamato, 13370 Sino-Japanese, 4421 foreign.
            # note that these counts differ from Takashi's counts. His method includes some words
            # that fail to be converted to IPA; manually inspecting reveals these words appear
            # to not have rendered correctly and appear just as boxes: ■


if __name__ == '__main__':
    main()
