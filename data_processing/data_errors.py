class UnsupportedKanaError(Exception):
    '''
    Error for kana my code doesn't support
    '''
    def __init__(self, kana):
        self.kana = kana

    def __str__(self):
        return 'Kana ' + self.kana + ' not supported'
