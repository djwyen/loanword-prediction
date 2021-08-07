'''
Quick script to process the data from the Balanced Corpus of Contemporary Written Japanese (BCCWJ), which is linked under the README.

This has some information on labels but it seems less relevant than the manual + Morita (2018)'s comments https://ccd.ninjal.ac.jp/bccwj/en/tsv.html
'''

import os
import csv
from data_processing import transcriber

# path to the short BCCWJ word frequency list
PATH_TO_TSV = "data/BCCWJ/BCCWJ_frequencylist_suw_ver1_0.tsv"

# we are interested in these particular properties:
#   lForm (index 1): a katakana transcription of the given word
#   lemma (index 2): the Japanese written form of the word, in kanji/hiragana/katakana
#   wType (index 5): the sublexicon of origin for a given word, namely native Yamato (和) Sino-Japanese (漢) or foreign (外).
#       There are also mixed etymology words (混), proper names (固), and symbols (記号),
#       which covers anything not written in a Japanese script (eg acronyms like 'DVD')
#   core_frequency (index 78): following Morita (2018), we only consider words with positive core_frequency, 
#       as it is only those words which were manually checked by humans for validity
YAMATO = '和'
SINO = '漢'
FOREIGN = '外'

def main():
    with open(PATH_TO_TSV) as f:
        reader = csv.reader(f, delimiter='\t')

        header = next(reader)
        # maps the name of a given property to its index in a data row for ease of use
        # the meaning of the properties is expounded upon in the BCCWJ manual, available at the same place as the actual data.
        # I can't actually read Japanese, but Morita's explanation of the corpus is sufficient for my purposes,
        #   and Google Translate + knowledge of 漢字 has served as a good aid in filling in the gaps
        prop_to_index = {header[i] : i for i in range(len(header))}

        # yamato_count = 0
        # sino_count = 0
        # foreign_count = 0
        # for entry in reader:
        #     core_freq = entry[prop_to_index['core_frequency']]
        #     if core_freq != '' and int(core_freq) > 0:
        #         origin_lang = entry[prop_to_index['wType']]
        #         if origin_lang==YAMATO:
        #             yamato_count += 1
        #         elif origin_lang==SINO:
        #             sino_count += 1
        #         elif origin_lang==FOREIGN:
        #             foreign_count += 1
        
        # print(f'{yamato_count} Yamato words')
        # print(f'{sino_count} Sino-Japanese words')
        # print(f'{foreign_count} Foreign words')
        # good, after applying the core frequency > 0 requirement we reproduce Takashi's counts (at least for SJ and foreign, which he discloses):
        # 9893 Yamato, 13373 Sino-Japanese, 4421 foreign

        t = transcriber.Transcriber()
        random_indices = {928, 123, 515, 209, 508, 40, 93, 320, 724, 540}
        for i in range(1000):
            x = next(reader)
            if i in random_indices:
                ipa = ''
                try:
                    ipa = t.katakana_to_ipa(x[1])
                except:
                    pass
                print(x[1], x[2], ipa)
                
            

if __name__ == '__main__':
    main()
