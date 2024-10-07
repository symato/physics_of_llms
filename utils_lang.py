import os, re, subprocess
from utils import LOCATION
import fasttext # pip install fasttext

## Fasttext detect lang
LANGID_FILENAME = 'lid.176.bin'
LANGID_FILEPATH = f"{LOCATION}/data/{LANGID_FILENAME}"

if not os.path.exists(LANGID_FILEPATH):
    cmd = f"wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/{LANGID_FILENAME}; mv {LANGID_FILENAME} {LANGID_FILEPATH}"
    subprocess.run(cmd, shell=True)

FASTTEXT_MODEL = fasttext.load_model(LANGID_FILEPATH)
WORD_RE = re.compile(r'\w+\s')

def detect_lang(text, check_words = False):
    if check_words:
        # Chỉ kiểm tra ngôn ngữ với những từ được phân tách rõ ràng
        words = re.findall(WORD_RE, text)
        text = " ".join(words)

    rs = FASTTEXT_MODEL.f.predict(text, 1, 0.0, 'strict')
    if rs: return rs[0][-1].split('__')[-1]
    else:  return None

assert detect_lang("hello vietnam") == "en"
assert detect_lang( "chào nước mỹ 123sfd http://adf4| tôi là ") == "vi"


_oa = ord('a')
_oz = ord('z')
_os = ord(' ')
###
def is_alphabet(token):
    for c in token.lower():
        o = ord(c)
        if o != _os and (_oa > o or o > _oz):
            return False
    return True

def mostly_alphabet(text):
    lower_text = text.lower()
    tokens = re.findall(r'[a-z]+', lower_text)
    n = 0
    for x in tokens:
        n += len(x)
    return n / len(text) > 0.8


import nltk # pip install nltk
nltk.download('words')
en_words = set( nltk.corpus.words.words() )
###
def is_english_word(token):
    x = token.strip().lower()
    if x in en_words: return True
    if len(x) > 1 and x[-1] == 's' and x[:-1] in en_words: return True # số nhiều
    return False

assert(is_english_word("car"))
assert(is_english_word("cars"))


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


import unicodedata
none_tone_syllables = set("khum toap chuyn mien khuech khun nun lin dong buan kay nhuyen ruet duet khoap lum nhit khoanh kheng lieng xuap thia luot truynh khoay thy quap vien chieng duya luech mai moam nguyech chuych phuan hach xuyen chum phoeng duyu giech suya loam choam bich biu phuya trai doa khim xom phec cunh buch quep rien meu muoi lunh uap penh xach chuang triec luo khi soi choo muu phun chuen meng oai nghenh toen que troam tap gheu cheu chuam rao khuynh nec phu ruom xuech thoac tuon hech bien ghi duyp ping khoi bap koai suyp thoam chieu gienh noe sap cau loap chay day pinh chuo cieu goao duon min nuyn mia nen nang thoao rech nam troanh lit toong ngha thuonh toac hooc tuat genh cup phia linh chich nhoai giua meo ngoat quyu chic xeu lep vic kat chuyech niet soeng khe hao nganh nhuon thau nuoch chuyenh tup tac boet piep vot choanh goam xec tuay doc dac bech oeo dang hoach loen set goon pooc noon toat phuyn nia nghau uyt kong nhoem kich bic xuyec coang nget thuam tuem khuyenh khoeo lien ging boo chot nghin voang man sut koa che quan uang moon ung hien nghit sun hue kien quai ghec hiu puya kuan ngoac goan suen thuy luenh tre dec moe thoo men ning quac soat nhim nuang nheo ngit ngheo nguyet bup doan pha nuot mach khoang bay thup nghienh troang hoang tem guyet nhiep phoai nung khoe hoai phuen kec oach ben koam ghech vuo beo choach phoanh quem thao giec giuong pic treng eng ngoe xoc huet khiem xoep roon nguen huonh xoe chinh ghet duang bum vich ngu niu phoem guo hiep chep ghip vuat beu tric chue kuop chat rim phoc goap lai piu noang bep moang tron luyp lia nge xenh chuet try soem duong guong trua huy vum ghem mon quing vuong dut dop giuoc hap viem nem vue luep ruem piet thiet tunh soc ron quiet khuam doe hoanh nghuoi cuya giai nieng giuc nhut phau phiec xuong guang pim puat sit vep pin rut nghinh rom nheng troeo diem kuoi xoang tai theu suou boch uyn van pen doat luc pang reo senh vunh sach liec quyenh phuyu bun huynh troi deng ret ghang xuom phy chiet quao sen pau tuep guech lac nghoanh giue booc roach toai doo nhuang khui nuyt hiem nhap vooc rang khuyn puong thoay mom vuya kao ceng minh quyet ram xien nhac chooc gen nau ngum guoi ghup thoat voon quong poac trop vuyet doang tuo soe vai buc nuat giom ren henh kuat nghom trinh duonh nac tren nhiet pong num pac guom loach chac ghep gioan luan poe son tet riep doen huon nhuc tinh muom cheng khuay nhin heo leo thec poc yen koay lonh gioay hich nguo phuoc xai mooc loanh xieng dim yem ruou vieng liet tuyt duen nay nguenh huay khet luong ghit uot dien truen dao hec pap poem nguan hoo gem conh phoang lop xuyet ong huenh nguong coo sum ghap thing nhien menh xem phim ghim nghac ruech kan nhet tuya hang quu rung chach xuyt uem muou noo phet tuyn chy nghang toeng hem tic tay piem pui but hoch nghe trum gioen uenh rum tat giao chuyec giuu khoeng nguyem ngieu much kiep nhuem tuang soach coan boi khuong gung bon ming khot toanh xia guc gium quia nhon gin phuyen thuynh phing oanh khoet choong khen oay choen trut nghut guych khuc uoi het hieu siec gut map quam giuam khit truyenh tuet uat phon kung khoat phuom chuom luyen nghuyen buon cat ngien nhuo triu vuoc dai nguem ngoot sip kenh tech troc noa ruen ngoec tuu ach cooc tom siem trin chuc rec giooc nom kin dieng coom khuen xen choap hung xop geo ranh phut puou huyu nghap nghao nich chuong xun nhoc nhoeng quang phot muop ngom choon huech manh koi nhech puu sot ruang siet khenh sop truan giong han phuc thuem giach buay san tuenh loac ngoa xiep buat huem quon nhiu giuyt lieu suap xoeo duynh nhui boac pien cuoc nghiem tung xao nhuen oen ghieu xan neo luynh pat pheu quech ruc buoc goeng khing munh cenh uoc mip vap puan nhoang chiec soang yet hienh seu khoem ganh miet khuop coem duan uong geng ich ham phem moem ngiu daa liu koang rin xuac huac guop nguu kanh roai seo ruan ghuan kam riet tuych koach voom tring voat buou thich roeng kuon nin lut nguay nguyen trup boom chit xuoc tieng nhoi noonh noam khiec thieu nhum vip giuop vem gheng ghop giay hep moo biep oem duc xip thenh song quoat quyng poo goach vun vuom boen huat duyenh kon nhenh quay xoo muonh diec nuou loet khuyen giuyn lem nghoc xinh gianh luyt loi suat giem khuenh kueng kut khuot mot hing let doem tenh thoang mienh vuec huoi suon phich tuoc sech ruch piec guem ruy lun uet chun sieu toep kim phup ngai oong soap ngoai sua keu bec kia ing truay ngy muot truou ria cun sing xanh nim quenh hay beng duoi gie khoep doi ngan nuop pao ngoach chuap giat lan deo xim hinh oat puyt nooc chuyt cim thuyet phiep tao gio troat mui noan troa poanh quoam nghech noen tooc cuo goe trieng moi xac nhuenh kooc dam hom lat roam thuyu loon sau ruat nginh dach suc nao truyu roi thuot ngho peu nghiep chonh hoe suyu ten leng vet bui tuou coai noao soao quen hoam nghiec trim quy khuom mua gooc nhoach put pue tuynh tuap cui xoec goat cuyen doam truoc hiec pun nghiu ving thoe gich rip vom ghien khu bat kic kuc ghan ruot nghat ken luay nhuu vui sui buoi loong reng boem toet bit khoach uech guyen ruoc them khiu roen nha soong gam dap dom cuoi choeo koong diu luoc bunh mam xing nhe huot kit lach binh ngen gian cung din gieng koem vin kai khanh toi gac puyu nguyeng och thuya dieu moanh ghiec khoo vao choom venh voen hau khoen guou khieng chec nhoet giuoi voan gan ghia khao voay ngoong mong nhach nguou kep huyet cot kui queo duech phip phinh tuong vec xit ghanh tuyenh bin pet dooc quim voi pit nghoeo huyep lap tut ruyn poay doao phuat duo dau dip vieu doch coa ngao chau hoon koom roo quoong hat buyet khooc heu ngia phooc mic nuc pia duat kap pum nghip tiep poai boat nuoi nua thoon trang quyeu hoeng ngang thi ngam noom gop oam gech nhoap muo sac mung duenh khia nup quach gao noi nhat truo xong khuyng luu chiem rooc bim hong khip king sem sep rua thuon pec oap punh chet phua gum nhieu due nghuy phuu ngooc vuy uya giap huyenh poen luyu chao vup muy kha dich hieng nan loom khuoi luap mue khat ngiem yep xiet nghi duyt xich nong khem kuong chuay hoao thuong siep goanh quuy loec phoong xiec khuu kienh roat puyn ghiu rieu goai cuu xuyenh dunh soac ngoen chan vuych tanh mum rep truyet chop phic nhoe giup nip loch phat huu vang ngui nue phuo khech khuyet thuyen nghen say mup sec voai ngoam truop roep moa nga xuc nghoi ngung yeu luy lam niem vong chai nuoc guat buyen gioong cing thuc soen dot thet cuon khueu tru luyet dem ngoi via nhom cen noay lot niec giang ninh thun vop buyu huan nhu phien chut duu khunh ngot nhoay ruyen phom xoam teng quau sat nghoe guy quet nghanh hop nghec huyen xup khap vung phum toe chuat xuych buang pay thoeng kheu nuyu nguot cuou biet khom nien tranh phay phuot quooc suec nieu hua trich giuyp trit kiet ngong nhem tip muc phep xuy thuyn xum nhang xoac koo nhoa nhoanh voe luych nuem thuech khien hoac goet truyp ngoang thot vech con khoec muyng miem xunh huang duyet chen ghong duch phang uyp ngin gioe boao anh trung xung lic luang tret xoa gang canh tan xang tang bom suoi long thuenh bot nghich thong ngech giu chi huyech xoat kua nguy hut doong vach nhoong hai tit uay thon kinh bam huong cac xiu troai gioai noat quynh hit phit xoan goong liem gioang doach phoam quinh pieng giau mep oec thien truc choeng guan vac vat kach nham khan nhuot nhoam nhuan ngiep loang guon poong noeng thieng hot pheo boon tien dua phach giui runh gap hic loay luet voanh den chanh thui buong ching boang gai choay loem nach troon phoo vim nuon huych trem poap xat tuop tri phong onh hun sien ghy huya pam tran vinh uep nhoat luop thuet tiu suyen duop hac thuoi cin nuyen bing gioanh ghiem vuou nghieng truyn trieu ngoet oon bet tray rem quoc tich nuyp phoi sim liep nhup trec bong triem gui xech choao teu kot lec keng quoan roao som lim nhuych yec soo puyet mop roan nhung bieng boanh khon phieng toeo xuam nhua leu then thut thep thuay len thoong phuyet cua guu nguech ech xui truyt hoan cinh trenh tua buop khy quat sic moong khonh nghia kheo duay pai kiem neu thiec bop roa quyn nuay hoc voa giop bia chua goang kac noong koe cuan gion huen ngich quin mao ngoem phop khai neng nhop oet coi phac qua suyn sue ngoc nghem sao ngieng thuych cec kang buyp get tuot thac choet mieng poang gioi luon gau nuech tronh khua goen viep nguop truych ngo det got guen mec gip phui pich heng nuya gioem voo sung giuom suech chem phai ghin khoon doom puc peo khoop hoet ruoi quoet nghien the nhunh quo hanh chunh trui nhue chup phieu luou xuang kuy nhoeo suop loao khuon tha xap nguon doet git trau soanh ghu cuong vam queng ton loe nguom nhiec choac chuenh nonh thiep veng pop nhun vuot vuu nech nhoan trep tuyu triep muya huam puom thoc tiem moai muyen vuen uyet xoeng monh khuang toon cop cien lung ghom gia troong tuen kun soet nuen hoay ruynh kuyen pieu cut viet oom xuon pho koc huop chong duych boa rot bieu xoai nhip cam nhieng boai xuay quoang cach ngoeo nguoi nho guoc thung bang muoc thuat muyn nhuop soai ghai phi dung nhy nhan hoap boach coon gheo von giuy suy tin tuy roang dech giut soon bue nghuong suay chuop sang tui muong khoai nghiet xep khop lau chuya cheo gho gha xua ngim diep lip xiem sieng lue nhanh luat trien suu lech goem goay huom ngoanh thoai luen phoen xuyp phuam nghy quuyet chung loc noc lui quyeng hoat kom chien luya rit sam ting nghic tuyep guyn tonh buot oeng uom gionh gic nuong khut hoen thang chuech xoon buyng cuc cuang xuyn hoep treo hin kho triet phoan xuya ngoao trip quop xut mieu poeng lang koanh khuoc nhay thoan duyn dup thoi guay khuyt nguoc phoe cham ghiep huyn khuyem keo viu giac vuyen xieu bao trom cao chuoc tue xoi khoao doeng thuen nhao vuop kue yeng not thic ciu choang ngay bienh doac xoom lich muat nghach nguat ngoay boeng ngoan uym choan trap phap ngoap suang pem chiep rai khic suyet pot thoen giuon boong roem lanh nguya huc guyu dit khieu phoa danh uan nop ran thonh khoa doon pheng sinh duot enh ghiet khuyech puo ngau ngeo nhoep nguych tuoi gep cuy gham nat soa doay truenh nhot trac gim loai coe cho siu pach rue nhoac diet xau giet nhuyet nhic mut xuenh tho chuec thai phuang choem uych thuap chuyem bai ric gioc tuam hui khung gieu nghung nhi voch khec kop xoap nguyt khuan vau xoet khuet phiem trot sup duem khang nuom gua may ngue ngheng roong ngheu uonh cuom ang hip nghieu gioao phung chang hoi choai nuy lao ceo uyen bac can xoay puoc tam cong muan nghep pip vut ruyt ooc phen guet khep ngun bem chuyep roet chu thuan khuych quoac xam trooc pan true nguyn chuyen trat vua rat mat nghu rieng sia uop nghan truon suan moan ciec buen suom soan poom thuoc vuoi nhep cap loeo troan viec thanh cha nhuom nenh giung phiu thoanh khuat uou doap guynh tim luoi quych toang nheu moc denh khuyec vuan voet kieng thay top gonh rinh xuynh cai kiec nghong coong vuon phuy truong xuen tot seng xoen thoeo nghuech cem phan chuu inh nuo quiu teo rong nguynh phiet khuyp cum xue ngut truot xic unh ruop phoac moeng oao gue ngi rop thim quoi nanh sooc truya duom roay vanh chuy run gong nhuoc bau veo nic nhen choc dui tuyem choat rac luom thum xuot chuynh riu gioa bua ruyu buu nhuoi choe gon toa quec lon thach qui vuc met puch quyem thop nep uoy mich chui nai tong kham lay xoanh toc ngop huo khoam nghuen chim nghing ghen soeo moach cuop quoay khach guyt goc luyn pung thuo tuyp dan luonh cay chuoi loat tau truu puyen guot thap dic xoong don vay tiec coc gay ruya ruon nguet sonh phoat koon ghot khinh thom boe khuo quich deu truom moac truat vuay ruong ghac quyen duyen gieo xuet thiem boan poam buyt ruenh ding xot mim xuan khue ghenh quym noai benh nuu banh hum toay puy tuyen donh dep tro nhuong moat khac mac vuet uen sin troen khoac ruyet rui pua suynh poon kip ngeu kieu lup loa nhai thoet puoi tia him phuem khuou miu buoy xuop xuch queu huap chom pom tun muyt ngua khuyu khoc tra noanh xet tria nui ven nhiem suong tach hia thuop cuot buo puon huyng ghun ngiet phonh chuon nguac choi bung quanh mun nhoen tiet duou trech kuyng thech vuem biem nuan gun renh cang ruych ghay nguc duam gec xoem loan panh treu goa lua chia sai quot tieu nghot hon buom gach buynh lenh rup coam ray truy bip vuyt tep thooc huou truyen guyp tuan ngap dinh guya muych roac koen koonh riec nhuynh roap kau nhia soam poa quit oac giuot ponh net ginh bach kuo uon toach that hoem suot vuang ngup phanh ruu chuan pham kiu hup suoc xeng ket khau xooc doanh nhec hoong cit muon dun goi khay mem pup rach moen rich rap ngach duoc uyu dia thuou buya com suam pech guym sich thuang nghim phao vit chenh ring voong nhich phech chiu nghai peng noet quip loo ghieng noem toem khich khuya tham nghuyet tum thua theo nuyet tuc ghach thoa coen thu thuu moom miec khoong khuy kem riem thue tuyet thinh oep chech than khoan looc ngham chuot oang soay cia duy thoach hunh loeng nap poat pep thuom nhuat roc room muen buy phuynh nut hoa moay trun toam khiep poi troe dat phuong phuon giuo puot tuom mang xoach biec mit gom nhinh tram noac phin xin quoanh ruo troac lom voc xuat voam ngac phenh ngenh chin nhuech non muay chuyet ling nguyu chip thip buyn toec ngat xuyu trong trach doai kech dum quoai roanh giam thit tec roe huyt tuyech goac xon boc ghat gien mech thiu nhuam khiet giun sanh theng nghet nhuay ghe choa xay oan cuat pon nhuyn quuyen xeo xuoi ban mau miep nhong truonh chon phuoi quyt too gup nhuyu hen kum thin giot truoi muet troet hiet nhuy nuynh ghich khup trao trunh goo huoc poan giep uam quuay khong phue niep khin rau ghinh nhooc uynh toan veu phe buet ngon nit tuech suyt gat suo xuu nhau reu chap gi ".strip().split())
def none_tone_vietnamese(s): # tieng_viet_khong_dau
    s = re.sub(u'Đ', 'D', s)
    s = re.sub(u'đ', 'd', s)
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore')
    return s.decode("utf-8")

def vietnamese_syllable_ratio(text):
    none_tone_text = none_tone_vietnamese(text)
    tokens = re.findall(r'[a-z]+', none_tone_text)

    if len(tokens) == 0:
        return 0

    syllable_count = 0

    for token in tokens:
        if token in none_tone_syllables:
            syllable_count += 1

    return syllable_count / len(tokens)

vi_chars = { 'ớ', 'ờ', 'ụ', 'ẫ', 'ổ', 'ậ', 'ẵ', 'â', 'ặ', 'ễ', 'ọ', 'ẩ', 'ỹ', 'ẽ', 'ủ', 'ạ', 'ấ', 'ư', 'ả', 'ỉ', 
'ỗ', 'ồ', 'ứ', 'đ', 'ự', 'è', 'ý', 'ế', 'ỵ', 'ũ', 'ắ', 'ẻ', 'ể', 'ợ', 'ệ', 'ẳ', 'ộ', 'à', 'õ', 'ĩ', 'ằ', 'ẹ', 'ỳ', 
'é', 'ử', 'ị', 'ở', 'ỡ', 'ê', 'ầ', 'ò', 'ề', 'ố', 'ỷ', 'ă', 'ì', 'ữ', 'ơ', 'ã', 'ỏ', 'ừ', 'ù', 'ú', 'á','ô', 'í', 'ó', 
'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y',
' ', '"', "'", ".", ",", ";", "_" }
# vi_word_re = re.compile(f'[0-9a-z{"".join(vi_chars)}]+', re.IGNORECASE)

def canbe_vietnamese(token):
    for c in token.lower():
        if c not in vi_chars:
            return False
    return True
'''
def canbe_vietnamese(token):
    count = 0
    for c in token.lower():
        if c in vi_chars:
            count += 1
    return count / len(token) >= 0.8
'''

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
有
县
は
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
