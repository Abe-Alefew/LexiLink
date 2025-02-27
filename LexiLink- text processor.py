
# importing useful libraries
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


# Reading the file and making it in string format
file_path_oromo = "parallel_text_oromo.txt"
file = open(file_path_oromo, 'r', encoding="utf8")
file_content_oromo = file.read()  # changing the txt format file into string format so that it will be easily processed later.

file_path_somali = "parallel_text_somali.txt"
file = open(file_path_somali, 'r', encoding="utf8")
file_content_somali = file.read()  # changing the txt format file into string format so that it will be easily processed later.


# Custom tokenization function to handle words with apostrophes
def custom_tokenize(text):
    pattern = re.compile(r"[\w']+")
    tokens = pattern.findall(text)
    processed_tokens = []
    for token in tokens:
        if "'" in token:
            # Special handling for tokens with apostrophes if needed
            processed_tokens.append(token)
        else:
            processed_tokens.append(token)
    return processed_tokens


# Main text preprocessing function





def text_preprocessing(text, language, lemmatizing_dict):
    # Removing extra white spaces
    combined_text = " ".join(text.split())

    # Tokenize the text using custom tokenizer
    tokens_of_the_text = custom_tokenize(combined_text)

    # Define stop words
    somali_stop_words = set([
        # Somali stop words as given
        'joogaa', 'waa maxay sababtu', 'banaanka', 'dhamaantood', 'ha noqon', 'adigaa tahay',
                         'kaasuu noqon doonaa', 'meesha', 'kor', 'isaga', 'naftaada', 'samee', 'noqonaya', 'badankood',
                         'ha noqonin', 'the', 'kuma jiro', 'dhax', 'yar', 'kaan', 'maya', 'qayb', 'halkaas', 'imisa',
                         'gudaha', 'dhinac', 'aad', 'uu', 'aniga', 'muhiim', 'hoosta', 'annaga', 'joojin', 'annagu',
                         'adigaa qabanaya', 'yaase', 'bidix', 'ayaa', 'kiinna', 'waa in uu ahaa', 'kiisa', 'waxaa',
                         'tani', 'sida', 'in', 'labadaba', 'wuxuu leeyahay', 'dibedda', 'kuwa', 'adiga', 'yaa', 'kale',
                         'ee', 'ka baxsan', 'kuwaas oo kale', 'yihiin', 'kiina', 'iyada', 'ama', 'ma', 'adigaa lahaanaya',
                         'doonaya', 'mid kasta', 'halkaan', 'oo', 'waxay', 'waxay leedahay', 'haddii', 'intaa ka badan',
                         'waa in', 'sameeyay', 'waxaan', 'marka', 'yahay', 'keebaa', 'kaas', 'horteeda', 'markaa', 'waqti',
                         'waxay ahayd', 'waa maxay', 'laakiin', 'ilaa', 'leeyahay', 'sidee', 'iyada tahay', 'halkee', 'ahaa',
                         'intaas ka dib', 'iska', 'naftiina', 'iyaga', 'aad u', 'hadda', 'kaligeed', 'kadib', 'kooda', 'ahaaday',
                         'waxba', 'naftooda', 'kaga', 'maxay', 'bixiya', 'na', 'keenna', 'dhabarka', 'maxaa dhacaya', 'soo',
                         'sidaa', 'kuwaas', 'waxay tahay', 'midig', 'kama', 'samaynaya', 'hoos', 'lahaa', 'isla', 'kan',
                         'waxa uu yahay', 'ahaanaya', 'ma aha', 'kayga', 'midna', 'ka', 'waa', 'maaha', 'isagu', 'kaasi',
                         'naftiisa', 'sar', 'kaliya', 'qaataa', 'adigaa haysta', 'kayaga', 'nafteeda', 'naftayda', 'midkoodna',
                         'ku', 'iyo', 'sababo', 'ay', 'horay', 'baxa', 'nafteena', 'kara'

    ])

    oromo_stop_words = set([
        # Oromo stop words as given
        'gatii', 'akka', 'bakka', 'irratti', 'bira', 'gara', 'keessatti',
    'duuba', 'fakkeessaa', 'jalaa', 'itti', 'gadii', 'ol', 'gubbaa',
    'biratti', 'irra', 'bitaa', 'mirga', 'qaama', 'ani', 'si', 'inni',
    'nuti', 'isaan', 'kana', 'isaa', 'maal', 'akkam', 'eenyu', 'eessa',
    'kan', 'nuyi', 'hin', 'kootu', 'keessani', 'isaa', 'keenya',
    'isaanii', 'ishee', 'jira', 'fakkaata', 'dhiyaata', 'fi', 'garuu',
    'yerro', "ta'u", 'natti', 'koo', 'keenyatu', 'keenyaa',
    'ati', "atiitu", "atiituu", "atituu", "atitu", 'keessa',
    'keessan', 'keessaniitu', 'isaaf', 'isaatu', 'ishoo',
    'ishee tu', 'ishee too', "innaatu", 'innata', 'inniitu', 'inni too',
    'isaaniitu', 'kam', 'eenyuu', 'eenyuun', 'kun', "kanaafu",
    'kunneen', 'kanneen', 'ta', 'akkii', 'akkuma', 'kama',
    'yoo', 'yookiin', 'sababa', 'yeroo', 'yeroo dura',
    'irra darbee', 'fuuldura', 'boodarra', 'gara gadii', 'gara gubbaa',
    'gara', 'gara keessa', 'gadaanaa', 'dabalata', 'erga',
    'yeroo tokko', 'as', 'achitti', 'maalif', 'hunduu', 'tokkuma',
    'lama', 'baayyee', 'cimsinee', 'tokko tokkoo', 'danda’aa', 'dha',
    'wanta', 'isa tokkicha', 'fkn', 'irra caalaa', 'kanaafuu',
    'garagalchitee', 's', 't', 'danda’a', 'ni', 'sirra', 'guutuu',
    'hin dhaga’amne', "hin", 'barbaadamuu', "hin barbaachisu",
    'amma', 'isa hin', 'hintaane', 'bu’aa hin', 'hin hojjetu',
    'yoo hin', "hin barbaachisu", "hin turre", 'hin jirtu',
    'ture', 'hin yaadatti', 'hin yaadamtu'
    ])

    # Select the stop words based on the language
    stop_words = somali_stop_words if language == 'somali' else oromo_stop_words

    # Filter out stop words
    filtered_tokens = [token for token in tokens_of_the_text if token.lower() not in stop_words]

    # Lemmatize the filtered tokens
    def lemmatize_text(tokens, lemmatizing_dict):
        lemmatized_tokens = [lemmatizing_dict.get(token, token) for token in tokens]
        return lemmatized_tokens

    lemmatized_tokens = lemmatize_text(filtered_tokens, lemmatizing_dict)

    # Joining tokens into a single string to apply regex and digit removal
    text = " ".join(lemmatized_tokens)

    # 1. Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)

    # 2. Remove digits using a loop
    reserve = ''
    num = '0123456789'
    for char in text:
        if char not in num:
            reserve += char
    text = reserve  # Assigning the cleaned text back to the text variable

    # Tokenize again if needed, or process the cleaned text further
    cleaned_tokens = text.split()

    # Converting text to lower case
    text = [word.lower() for word in cleaned_tokens]

    # Return the processed text
    return " ".join(text)


# lemmatizing dictionary for Oromo
oromic_lemmatizing_dict = {
    # Nouns
    'buufataalee': 'buufata', 'waraanaa': 'waraana', 'Huutii': 'Huutii', 'haleellaa': 'haleellaa',
    'qilleensarraa': 'qilleensa', 'beeksifte': 'beeksif', 'magaala': 'magaala', 'doonii': 'doonii',
    'Hodeedaa': 'Hodeedaa', 'halellaan': 'haleellaa', 'Rooyitarsitti': 'Rooyitars', 'humni': 'humna',
    'miidiyaa': 'miidiyaa', 'tarkaanfii': 'tarkaanfii', 'hidhattoonni': 'hidhatto', 'Teel Aviiv': 'Teel Aviiv',
    'abiddi': 'abidi', 'Baha Giddu Galeessaa': 'Baha Giddu Galeessaa', 'samiitti': 'sami', 'suuraleen': 'suura',
    'Mootummaan': 'Mootummaa', 'kuusaa': 'kuusaa', 'boba’aafi': 'boba’a', 'ibsa': 'ibsa', 'namoonni': 'namo',
    'nagaan': 'nagaa', 'miidhaman': 'miidham', 'lammii': 'lammii', 'dogoggora': 'dogoggora', 'Hamaas': 'Hamaas',
    'USfi': 'US', 'UKn': 'UK', 'qiyyaafannoo': 'qiyyaafannoo', 'cinaa': 'cinaa', 'Haleellan': 'haleellaa',
    'Adoolessa': 'Adoolessa', 'Hodeedaa': 'Hodeedaa', 'Rooyitars': 'Rooyitars', 'miidiyaa': 'miidiyaa',
    'Sana’a': 'Sana’a', 'Galaana': 'Galaana', 'Diimarratti': 'Diimarratti', 'boba’a': 'boba’a', 'ibsa': 'ibsa',
    'Haleellaa': 'haleellaa', 'suuraleen': 'suura', 'Ministiirri': 'Ministiirri', 'Yaa’oov': 'Yaa’oov',
    'Gaalant': 'Gaalant', 'Huutiin': 'Huutii', 'Filisxeemota': 'Filisxeemota', 'Gaazaa': 'Gaazaa',
    'abiddaa': 'abiddaa', 'suuralen': 'suura', 'Tiwitara': 'Tiwitara', 'qiyyaafannoowwan': 'qiyyaafannoo',
    'shororkeessaa': 'shororkeessaa', 'Hodeedaa': 'Hodeedaa', 'Mootummaan': 'Mootummaan', 'Haaloo': 'Haaloo',
    'baha': 'baha', 'Giddu': 'Giddu', 'Galeessaa': 'Galeessaa', 'magaala': 'magaala',

    # Verbs
    'gaggeessuu': 'gaggeess', 'gaggeeffame': 'gaggeeff', 'hima': 'him', 'fudhachuuf': 'fudhachu',
    'gaggeessite': 'gaggeess', 'dubbatan': 'dubbatu', 'raawwate': 'raawwat', 'deeggare': 'deegg',
    'beeksise': 'beeksis', 'fayyadamuun': 'fayyad', 'qolachuu': 'qolach', 'miidhan': 'miidh',
    'kuffistee': 'kuffis', 'dhiibbaa': 'dhiibba', 'beeksisaa': 'beeksis', 'gaggeeffame': 'gaggeeff',
    'raawwate': 'raawwat', 'dirqamte': 'dirqam', 'fudhachuuf': 'fudhachu', 'deebii': 'deebii',
    'ya’aan': 'ya’a', 'hidhattoonni': 'hidhatto', 'dubbatan': 'dubbatu', 'taate': 'ta',
    'raawwate': 'raawwat', 'gaggeessaa': 'gaggeess', 'miidhu': 'miidh', 'agarsiisu': 'agarsiis',

    # Adjectives
    'xiyyaaronni': 'xiyyaar', 'heddu': 'heddu', 'cimaa': 'cimaa', 'guddaan': 'gudda',
    'guddaa': 'gudda', 'hin': 'hin', 'gargaarsa': 'gargaars', 'naannoo': 'naannoo', 'gubbaa': 'gubba',
    'kan': 'kan',
}

somalic_lemmatizing_dict = {
    # Nouns
    'israaiil': 'israaiil', 'duqaymo': 'duqaymo', 'cirka': 'cirka', 'saldhig': 'saldhig',
    'milatariga': 'milatarig', 'xuuthiyiinta': 'xuuthiyiint', 'yemen': 'yemen',
    'ciidanka': 'ciidank', 'bogga': 'bog', 'diyaaradaha': 'diyaarad', 'dagaalka': 'dagaal',
    'tel_aviv': 'tel_aviv', 'masuuliyadda': 'masuuliy', 'duuliye': 'duuliye', 'kala': 'kala',
    'khalad': 'khalad', 'hantiwadaag': 'hantiwadaag', 'midowga': 'midowga', 'soofiyeeti': 'soofiyeeti',
    'magaca': 'magac', 'iskoole': 'iskool', 'heeso': 'hees', 'suurad': 'suurad',
    'dhinaca': 'dhinac', 'soomaaliya': 'soomaaliya', 'dhexe': 'dhexe', 'xafiiska': 'xafiis',
    'ururada': 'urur', 'bulshada': 'bulsh', 'oromia': 'oromia', 'magaalo': 'magaal',
    'adeeg': 'adeeg', 'shaqo': 'shaqo', 'xiisad': 'xiisad', 'saameyn': 'saameyn',
    'xusuusta': 'xusuus', 'maydka': 'mayd', 'muraayad': 'muraayad', 'dhagxaan': 'dhagxaan',
    'shaqaalaha': 'shaqaale', 'xuska': 'xus', 'faallo': 'faallo', 'taariikh': 'taariikh',
    'sida': 'sida', 'daweynta': 'daweyn', 'wasiir': 'wasiir', 'soofiyeet': 'soofiyeet',
    'soomaali': 'soomaali', 'saldhigyada': 'saldhig', 'iska': 'iska',
    'guri': 'guri', 'bariga': 'barig', 'maamulka': 'maamul', 'qeexitaan': 'qeex',
    'mashruuc': 'mashruuc', 'shaandheyn': 'shaandheyn', 'dayactir': 'dayactir',
    'dhawaaq': 'dhawaaq', 'gaar': 'gaar', 'deegaan': 'deegaan',

    # Verbs
    'ku_hayso': 'ku_hay', 'ku_shaaciyay': 'ku_shaaci', 'ka_fuliyeen': 'ka_fuli', 'weerar': 'weerar',
    'qaadeen': 'qaad', 'sheegtay': 'sheeg', 'dilay': 'dil', 'dhaawacmeen': 'dhaawac',
    'sheegeen': 'sheeg', 'bixiyaan': 'bixi', 'hubiyo': 'hubi', 'sii': 'si', 'maamuus': 'maamuus',
    'sameeyay': 'samee', 'dhacday': 'dhac', 'qabsaday': 'qabso', 'soomaal': 'soomaal', 'dhiibay': 'dhiib',
    'maamul': 'maamul', 'qoray': 'qor', 'buuxiyay': 'buuxi', 'qaadayo': 'qaad', 'weeyaan': 'weey',
    'saareen': 'saare', 'xil': 'xil', 'dhigay': 'dhig', 'dhibaan': 'dhib',
    'soo_saar': 'soo_saar', 'qaateen': 'qaad', 'xakameyn': 'xakameyn', 'heshiis': 'heshiis',
    'fahmay': 'fahm', 'wada_hadlay': 'wada_hadl', 'bixiyay': 'bix', 'caddeeyay': 'caddeey',
    'dayactirayaa': 'dayactir', 'u_huleeyay': 'u_hule', 'jirta': 'jir', 'fiiqay': 'fiiq',
    'haddii': 'haddii', 'dhiibayaan': 'dhiib', 'siisaan': 'siis', 'ku_hay': 'ku_hay',
    'galay': 'gala', 'wada': 'wada',

    # Adjectives
    'culus': 'culus', 'xun': 'xun', 'gaar': 'gaar', 'farsamo': 'farsamo', 'xaqiiqda': 'xaqiiq',
    'wayn': 'wayn', 'guul': 'guul', 'qaalisan': 'qaalis', 'iska': 'iska', 'halis': 'halis',
    'daran': 'daran', 'hufan': 'hufan', 'degdeg': 'degdeg', 'mugdi': 'mugdi', 'hagaagsan': 'hagaagsan',
    'mugdi': 'mugdi', 'waxa': 'waxa', 'kaab': 'kaab', 'sar': 'sar', 'iska': 'iska', 'miisaan': 'miisaan',
    'xakameyn': 'xakameyn', 'xagjir': 'xagjir', 'daweyn': 'daweyn', 'xidhiidh': 'xidhiidh',
    'qoto dheer': 'qoto dheer',
}

# Preprocess the texts
preprocessed_text_oromo = text_preprocessing(file_content_oromo, 'oromo', oromic_lemmatizing_dict)
preprocessed_text_somali = text_preprocessing(file_content_somali, 'somali', somalic_lemmatizing_dict)

print("Afan Oromo pre-processed text is:\n", preprocessed_text_oromo)
print("")
print("Somali pre-processed text is:\n", preprocessed_text_somali)


def tokenizer(text):
    words = text.split()
    return words

#For further analysis of text processing, we need to tokenize each words of the cleaned text
preprocessed_text_oromo = tokenizer(preprocessed_text_oromo)
preprocessed_text_somali = tokenizer(preprocessed_text_somali)


# Counting frequency of words

def word_counter(text, type):
    word_count = Counter(text)
    print(f"The frequency of the word in the given {type} text: ")
    for word, count in word_count.items():
        print(f"The {word} word in {type} text: {count}x")  # returning how many times does the word happen in the text
    print("-----------------------------------------------------------------------")
    # creating a new file and writing the word frequencies on it
    f_text = open(f"word frequency {type}.txt", 'w', encoding="utf-8")
    for word, count in word_count.items():
        f_text.write(f"The {word} word in {type} text: {count}x\n")


word_counter(preprocessed_text_oromo, 'oromo')
word_counter(preprocessed_text_somali, 'somali')


# calculating the overlap between the common words in the parallel texts of oromo and somali
def overlap(words_1, words_2):
    set_words1 = set(words_1)
    set_words2 = set(words_2)
    ovrlap = set_words1 & set_words2
    # putting the frequency of the common words in percentage
    overlap_percentage = (len(ovrlap) / ((len(set_words1) + len(set_words2)) / 2)) * 100
    print(f"Common words({len(ovrlap)}): {ovrlap}")
    return print(f"Overlap percentage: {overlap_percentage:.2f}%")


overlap(preprocessed_text_somali, preprocessed_text_oromo)

print('-----------------------------------------------------------------------')

# the above words are in grapheme form. we will now transcribe them into phenomes form

# Creating phenomes dictionary
afan_oromo_phoneme_dict = {
    'a': 'a',
    'b': 'b',
    'c': 'ʧ',  # 'ch' sound
    'd': 'd',
    'dh': 'ɖ',  # Retroflex stop
    'e': 'e',
    'f': 'f',
    'g': 'g',
    'h': 'h',
    'i': 'i',
    'j': 'ʤ',  # 'j' sound
    'k': 'k',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'o': 'o',
    'p': 'p',
    'q': 'q',
    'r': 'r',
    's': 's',
    'sh': 'ʃ',  # 'sh' sound
    't': 't',
    'u': 'u',
    'w': 'w',
    'x': 'x',
    'y': 'j',  # 'y' sound
    'z': 'z',
    'aa': 'aː',  # Long 'a'
    'ee': 'eː',  # Long 'e'
    'ii': 'iː',  # Long 'i'
    'oo': 'oː',  # Long 'o'
    'uu': 'uː',  # Long 'u'
    'ts': 'ts',  # Affricate sound
    'ph': 'pʰ',  # Aspirated 'p'
    'th': 'tʰ',  # Aspirated 't'
    'kh': 'kʰ',  # Aspirated 'k'
    'ng': 'ŋ',  # Nasal sound
    'ny': 'ɲ',  # Palatal nasal
    'hw': 'ʍ',  # Voiceless labio-velar fricative
    'j': 'ʤ',  # 'j' sound
    'ch': 'ʧ',  # 'ch' sound
}

somali_phoneme_dict = {
    'a': 'a',  # Open front unrounded vowel
    'b': 'b',  # Voiced bilabial plosive
    'c': 'ʧ',  # Voiceless postalveolar affricate (similar to 'ch')
    'd': 'd',  # Voiced alveolar plosive
    'e': 'e',  # Close-mid front unrounded vowel
    'f': 'f',  # Voiceless labiodental fricative
    'g': 'g',  # Voiced velar plosive
    'h': 'h',  # Voiceless glottal fricative
    'i': 'i',  # Close front unrounded vowel
    'j': 'ʤ',  # Voiced postalveolar affricate
    'k': 'k',  # Voiceless velar plosive
    'l': 'l',  # Voiced alveolar lateral approximant
    'm': 'm',  # Voiced bilabial nasal
    'n': 'n',  # Voiced alveolar nasal
    'o': 'o',  # Close-mid back rounded vowel
    'p': 'p',  # Voiceless bilabial plosive
    'q': 'q',  # Voiceless uvular plosive
    'r': 'r',  # Voiced alveolar trill
    's': 's',  # Voiceless alveolar fricative
    'sh': 'ʃ',  # Voiceless postalveolar fricative (similar to 'sh')
    't': 't',  # Voiceless alveolar plosive
    'u': 'u',  # Close back rounded vowel
    'v': 'v',  # Voiced labiodental fricative
    'w': 'w',  # Voiced labio-velar approximant
    'x': 'x',  # Voiceless velar fricative
    'y': 'j',  # Voiced palatal approximant
    'z': 'z',  # Voiced alveolar fricative
    'aa': 'aː',  # Long open front unrounded vowel
    'ee': 'eː',  # Long close-mid front unrounded vowel
    'ii': 'iː',  # Long close front unrounded vowel
    'oo': 'oː',  # Long close-mid back rounded vowel
    'uu': 'uː',  # Long close back rounded vowel
}


# G2P conversion function
def g2p_converter(text, phoneme_dict):
    phonemes = []
    for word in text:
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i:i + 2] in phoneme_dict:
                phonemes.append(phoneme_dict[word[i:i + 2]])
                i += 2
            else:
                phonemes.append(phoneme_dict.get(word[i], word[i]))
                i += 1
    return phonemes


phonemes_afan_oromo = g2p_converter(preprocessed_text_oromo, afan_oromo_phoneme_dict)
phonemes_somali = g2p_converter(preprocessed_text_somali, somali_phoneme_dict)
print(f"Phonemes of the afan oromo text: {phonemes_afan_oromo}")
print(f"Phonemes of somali text: {phonemes_somali}")

print('---------------------------------------------------------------------------------------------')
f_text = open(f"transcribed phonemes of afan oromo words.txt", 'w', encoding="utf-8")
for word in preprocessed_text_oromo:
    f_text.write(f"{word}, an afan oromo word,has a phoneme form of {g2p_converter(word, afan_oromo_phoneme_dict)}\n")
f_text = open(f"transcribed phonemes of somali words.txt", 'w', encoding="utf-8")
for word in preprocessed_text_somali:
    f_text.write(f"{word}, a somali word,has a phoneme form of {g2p_converter(word, somali_phoneme_dict)}\n")
# Using the bulitin function of counter for counting the frequencey of the pheonmes

frequency_afan_oromo = Counter(phonemes_afan_oromo)
frequency_somali = Counter(phonemes_somali)
for word, count in frequency_afan_oromo.items():
    print(f"The frequency of {word},an afan oromo phoneme, in the afan oromo parallel text :{count}")
print("-----------------------------------------------------------------------")
for word, count in frequency_somali.items():
    print(f"The frequency of {word},a somali phoneme, in the somali parallel text :{count}")
print("-----------------------------------------------------------------------")


# Normalize frequencies
def normalize_frequencies(frequency_counter):
    total_count = sum(frequency_counter.values())
    return {phoneme: (count / total_count) for phoneme, count in frequency_counter.items()}


normalized_frequency_afan_oromo = normalize_frequencies(frequency_afan_oromo)
normalized_frequency_somali = normalize_frequencies(frequency_somali)


# Compute overlap
def compute_overlap(frequency1, frequency2):
    common_phonemes = set(frequency1.keys()) & set(frequency2.keys())
    overlap = {phoneme: (frequency1[phoneme], frequency2[phoneme]) for phoneme in common_phonemes}
    return overlap


phoneme_overlap = compute_overlap(normalized_frequency_afan_oromo, normalized_frequency_somali)

# Display the results
print("Phoneme Overlap:")
for phoneme, (freq1, freq2) in phoneme_overlap.items():
    print(f"Phoneme: {phoneme}, Afan Oromo Frequency: {freq1:.2%}, Somali Frequency: {freq2:.2%}")

# plotting

# Extracting data for plotting
phonemes = list(phoneme_overlap.keys())
afan_oromo_freq = [freq[0] for freq in phoneme_overlap.values()]
somali_freq = [freq[1] for freq in phoneme_overlap.values()]

# Plotting the bar graph
x = np.arange(len(phonemes))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, afan_oromo_freq, width, label='Afan Oromo')
rects2 = ax.bar(x + width / 2, somali_freq, width, label='Somali')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Phonemes')
ax.set_ylabel('Frequency')
ax.set_title('Phoneme Frequency Overlap')
ax.set_xticks(x)
ax.set_xticklabels(phonemes)
ax.legend()
fig.tight_layout()
plt.show()
