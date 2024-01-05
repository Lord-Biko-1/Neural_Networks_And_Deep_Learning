from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from textblob import Word
import warnings
import emoji
import re
from deep_translator import GoogleTranslator
warnings.simplefilter("ignore")


class Preprocess:

    def __init__(self,data):
        self.data = data

    def translate_text(text):
        # print("1",text)
        english_text = re.sub("[^A-Za-z ]", "", str(text))
        if english_text.isspace():
            return text
        else:
            print("2", english_text)
            translate = GoogleTranslator(source="auto", target="arabic").translate(english_text)
            words = text.split(' ')
            translated_words = []
            for word in words:
                if re.match("[\u0600-\u06FF]", word):
                    translated_words.append(word)
            translated_words.append(translate)
            translation = ' '.join(translated_words)
            print("3", translation)
            return translation

    # Function to preprocess text
    def preprocess_text(self):
        # convert emoji to text
        self.data = self.data.apply(lambda x: " ".join(' ' + word + ' ' for word in x.split()))
        self.data = self.data.apply(lambda x: " ".join(emoji.demojize(word) for word in x.split()))

        # remove everything except words and numbers
        reg = r'[^\w\s]'
        self.data = self.data.replace(reg, ' ', regex=True)
        self.data = self.data.replace('_', ' ', regex=True)

        # translate english to arabic
        # self.data = self.data.apply(lambda x: translate_text(x))

        # make the sentences  in lower case
        self.data = self.data.apply(lambda x: " ".join(x.lower() for x in x.split()))

        # remove arabic stop words
        stop = stopwords.words('arabic')
        self.data = self.data.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

        # make arabic stemming
        st = ISRIStemmer()
        self.data = self.data.apply(lambda x: " ".join([st.suf32(word) for word in x.split()]))

        # removing stop wards from the data
        stop = stopwords.words('english')
        self.data = self.data.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

        # apply lemmatization in the data
        self.data = self.data.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

        return self.data