def segment_len_of_ipa(word_in_ipa):
    """
    A Japanese-specific method for calculating the length of a word in IPA.
    In particular also specific to my BROAD transcription of Japanese.
    This lets one calculate lengths of words (eg for padding sequences)
    far faster by using Python built in functions instead of costly
    calls to panphon to transform ipa into feature vectors then calculate the length of that array.
    """
    length = len(word_in_ipa)
    # get rid of characters that are really diacritics of the previous segment
    length -= word_in_ipa.count('ː') + word_in_ipa.count('ʲ')
    # correct for characters counted as two segments due to their unicode representations
    length -= word_in_ipa.count('ç') + word_in_ipa.count('ɰ̃') + word_in_ipa.count('ĩ')
    # correct for affricates, which are one segment counted as three (two segs + joiner)
    length -= 2 * ( word_in_ipa.count('d͡ʑ') + word_in_ipa.count('d͡z') + word_in_ipa.count('t͡ɕ') + word_in_ipa.count('t͡s') )
    return length
