import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Input, Embedding, Dense, Dropout, MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from textblob import Word
from keras.preprocessing.sequence import pad_sequences
import warnings
import keras
import emoji
import re
from deep_translator import GoogleTranslator
warnings.simplefilter("ignore")


def translate_text(text):
    # print("1",text)
    english_text = re.sub("[^A-Za-z ]", "", str(text))
    if english_text.isspace():
        return text
    else:
        # print("2",english_text)
        translate = GoogleTranslator(source="auto", target="arabic").translate(english_text)
        words = text.split(' ')
        translated_words = []
        for word in words:
            if re.match("[\u0600-\u06FF]", word):
                translated_words.append(word)
        translated_words.append(translate)
        translation = ' '.join(translated_words)
        # print("3",translation)
        return translation


# Function to preprocess text
def preprocess_text(Data):
    # convert emoji to text
    Data = Data.apply(lambda x: " ".join(' ' + word + ' ' for word in x.split()))
    Data = Data.apply(lambda x: " ".join(emoji.demojize(word) for word in x.split()))

    # remove everything except words and numbers
    reg = r'[^\w\s]'
    Data = Data.replace(reg, ' ', regex=True)
    Data = Data.replace('_', ' ', regex=True)

    # translate english to arabic
    # Data = Data.apply(lambda x: translate_text(x))

    # make the sentences  in lower case
    Data = Data.apply(lambda x: " ".join(x.lower() for x in x.split()))

    # remove arabic stop words
    stop = stopwords.words('arabic')
    Data = Data.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    # make arabic stemming
    st = ISRIStemmer()
    Data = Data.apply(lambda x: " ".join([st.suf32(word) for word in x.split()]))

    return Data

# Read data
data = pd.read_csv("train.CSV")
y = data["rating"]
zero,one,two = 0,0,0
for i in range(len(y)):
    if y[i] == -1:
        y[i] = 0
        zero = zero + 1
    elif y[i] == 0:
        y[i] = 1
        one = one + 1
    if y[i] == 1:
        y[i] = 2
        two = two + 1
# print(zero)
# print(one)
# print(two)
# print(y)

X = data["review_description"]
X = preprocess_text(X)
print(X[44])
print(X[54])
print(X)

# Tokenize the words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
# print(X_sequences)

# Pad sequences to make them of the same length
X_padded = pad_sequences(X_sequences)
# print(X_padded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42, shuffle=True)

# Define the model
# model = keras.Sequential()

inputs = Input(shape=(X_train.shape[1],))
x = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128)(inputs)
print(x)

# Transformer Block 1
attn1 = MultiHeadAttention(num_heads=8, key_dim=128)(x, x)
attn1 = Dropout(0.1)(attn1)
# attn = Add()
out1 = LayerNormalization(epsilon=1e-6)(x + attn1)

# Transformer Block 2
attn2 = MultiHeadAttention(num_heads=8, key_dim=128)(out1,out1)
attn2 = Dropout(0.1)(attn2)
# model.add(Add())
out2 = LayerNormalization(epsilon=1e-6)(out1 + attn2)

# FeedForward Layer
ffn = Dense(units=128, activation='relu')(out2)
ffn = Dropout(0.1)(ffn)
# model.add(Add()
out3 = LayerNormalization(epsilon=1e-6)(out2 + ffn)

# Global Average Pooling
out3 = GlobalAveragePooling1D()(out3)
out3 = Dense(32, activation='relu')(out3)
outputs = Dense(3, activation='softmax')(out3)

model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
opt = keras.optimizers.Adam(learning_rate=0.001, clipvalue=5.0)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)  # You can adjust the momentum value
# model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=2)

# Train the model
num_epochs = 5
history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), verbose=2,callbacks=[es])

# Test the model
data_test = pd.read_csv("test _no_label.csv")
test = data_test["review_description"]
test = preprocess_text(test)

# Tokenize the words
test_sequences = tokenizer.texts_to_sequences(test)
# print(len(test_sequences))

# Pad sequences to make them of the same length
test_padded = pad_sequences(test_sequences)


new_test_padded = []
for i in range(0,1000):
    temp = np.array(test_padded[i])
    zeros_to_pad = max(0, X_train.shape[1] - len(temp))
    pad_before = zeros_to_pad // 2
    pad_after = zeros_to_pad - pad_before
    temp = np.pad(temp, (pad_before, pad_after), mode='constant', constant_values=0)
    new_test_padded.append(temp)

new_test_padded = np.array(new_test_padded)
print(f"test pad1{new_test_padded.shape}")

predict = model.predict(new_test_padded)
# print(len(predict))
final = []
for i in range(len(predict)):
    maximum = max(predict[i])
    if maximum == predict[i][0]:
        final.append(-1)
    elif maximum == predict[i][1]:
        final.append(0)
    elif maximum == predict[i][2]:
        final.append(1)

print(final)

output = pd.DataFrame({'ID': data_test.ID, 'rating': final})
output.to_csv('submission1.csv', index=False)
print("Your submission was successfully saved!")