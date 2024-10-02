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

def is_ascii(token):
    for char in token:
        if ord(char) > 255:
            return False
    return True

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

unwanted_langs_re = regex.compile(f'[{"".join(unwanted_langs)}]+')

unwanted_lang_re_pairs = {}

for x in unwanted_langs:
    unwanted_lang_re_pairs[x[3 : -1]] = regex.compile(f'[{x}]+')
# print(unwanted_lang_re_pairs)

def check_check(reg, token):
    m = regex.findall(reg, token)
    for x in m:
        for c in x:
            if ord(c) > 255: # not ascii
                return True
    return False


def write_to_lang_file(lang, token):
    filename = f"data/langs/{lang}.txt"
    with open(filename, "at") as f:
        f.write(token + "\n")


def check_lang(token):
    belongs_to_at_least_one_lang = False

    for lang, reg in unwanted_lang_re_pairs.items():
        if check_check(reg, token):
            belongs_to_at_least_one_lang = True
            write_to_lang_file(lang, token)

    if not belongs_to_at_least_one_lang:
        if contains_cjk(token):
            write_to_lang_file("CJK", token)
        else:
            write_to_lang_file("Others", token)



# https://emoji-python.readthedocs.io/en/stable/
from emoji import emoji_count # python -m pip install emoji --upgrade

def contains_emoji(token):
    return emoji_count(token) > 0

def contains_unwanted(token):
    if contains_cjk(token):
        return True

    m = regex.findall(unwanted_langs_re, token)
    for x in m:
        for c in x:
            if ord(c) > 255: # not ascii
                return True
    return False


vi_chars = {'ớ', 'ờ', 'ụ', 'ẫ', 'ổ', 'ậ', 'ẵ', 'â', 'ặ', 'ễ', 'ọ', 'ẩ', 'ỹ', 'ẽ', 'ủ', 'ạ', 'ấ', 'ư', 'ả', 'ỉ', 'ỗ', 'ồ', 'ứ', 'đ', 
'ự', 'è', 'ý', 'ế', 'ỵ', 'ũ', 'ắ', 'ẻ', 'ể', 'ợ', 'ệ', 'ẳ', 'ộ', 'à', 'õ', 'ĩ', 'ằ', 'ẹ', 'ỳ', 'é', 'ử', 'ị', 'ở', 'ỡ', 'ê', 
'ầ', 'ò', 'ề', 'ố', 'ỷ', 'ă', 'ì', 'ữ', 'ơ', 'ã', 'ỏ', 'ừ', 'ù', 'ú', 'á', 'ô', 'í', 'ó',
'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', ' ',
'"', "'", ".", ",", ";" }

def canbe_vietnamese(token):
    for c in token.lower():
        if c not in vi_chars:
            return False
    return True

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

min_cjk = 11935
# min_cjk = ord('\u31c0')

max_cjk = 64055
# max_cjk = ord('\u9fff')

if __name__ ==  "__main__":

    unwanted = """
ทรูวิชั่นส์asdf, ầds tiến lên
게시판
활
⽗
臘
怒
辰
⺟
旅
里
拓
見
嘆
""".strip().split("\n")

    for x in unwanted:
        if not contains_unwanted(x):
         print(x)
         print(min_cjk, max_cjk)
         for c in x:
            print(ord(c), c)


    emoji_samples = """
🈯
🈲
🈹
🌇
🌓
🍘
🎑
🎿
🏏
🏒
🏩
🏯
🐀
👝
💹
💺
📟
📪
📼
🔀🔂
🔃
🔇
🔓
🔢
🔤🔩
🕖
🕚
🕜
🕝
🕞
🕠🕢
🌍,
 😂, 😃,
 😂
    """.strip().split("\n")
    
    for x in emoji_samples:
        if not contains_emoji(x):
            print(x, emoji_count(x))


    vi_samples = """
    hê hê
    an
    " VIỆT

    """.strip().split("\n")

    for x in vi_samples:
        if not canbe_vietnamese(x):
            print(x)
