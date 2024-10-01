'''
https://www.regular-expressions.info/unicode.html
https://stackoverflow.com/questions/38615740/regular-expression-to-accept-all-thai-characters-and-english-letters-in-python#answer-72440821
https://pypi.org/project/regex/
'''
import regex

def contains_cjk(token):
    for char in token:
        o = ord(char)
        if min_cjk <= o and o <= max_cjk:
            return True
    return False

def not_ascii(token):
    for char in token:
        if ord(char) > 255:
            return True
    return False

unwanted_langs = '''
\p{Arabic}
\p{Armenian}
\p{Bengali}
\p{Bopomofo}
\p{Braille}
\p{Buhid}
\p{Cherokee}
\p{Cyrillic}
\p{Devanagari}
\p{Ethiopic}
\p{Georgian}
\p{Greek}
\p{Gujarati}
\p{Gurmukhi}
\p{Hanunoo}
\p{Hebrew}
\p{Hiragana}
\p{Inherited}
\p{Kannada}
\p{Katakana}
\p{Khmer}
\p{Lao}
\p{Limbu}
\p{Malayalam}
\p{Mongolian}
\p{Myanmar}
\p{Ogham}
\p{Oriya}
\p{Runic}
\p{Sinhala}
\p{Syriac}
\p{Tagalog}
\p{Tagbanwa}
\p{TaiLe}
\p{Tamil}
\p{Telugu}
\p{Thaana}
\p{Thai}
\p{Tibetan}
\p{Yi}
'''.strip().split()

unwanted_langs = "".join(unwanted_langs)
unwanted_langs_re = regex.compile(f'[{unwanted_langs}]+')


def contains_unwanted(token):
    if contains_cjk(token):
        return True

    m = regex.findall(unwanted_langs_re, token)
    for x in m:
        for c in x:
            if ord(c) > 255: # not ascii
                return True
    return False


'''
The 4E00—9FFF range covers CJK Unified Ideographs (CJK=Chinese, Japanese and Korean). 
There are a number of lower ranges that relate, to some degree, to CJK:

31C0—31EF CJK Strokes
31F0—31FF Katakana Phonetic Extensions
3200—32FF Enclosed CJK Letters and Months
3300—33FF CJK Compatibility
3400—4DBF CJK Unified Ideographs Extension A
4DC0—4DFF Yijing Hexagram Symbols
4E00—9FFF CJK Unified Ideographs 
'''
# min_cjk = ord('\u31c0')
min_cjk = ord('\u4e00')
# max_cjk = ord('\u9fff')
max_cjk = 54876

if __name__ ==  "__main__":

    samples = """
ทรูวิชั่นส์asdf, ầds tiến lên
게시판
활
""".strip().split("\n")

    for x in samples:
        if not contains_unwanted(x):
         print(x)
         print(min_cjk, max_cjk)
         for c in x:
            print(ord(c), c)

'''

python3 tokens_check.py > latin_tokens.txt

'''
