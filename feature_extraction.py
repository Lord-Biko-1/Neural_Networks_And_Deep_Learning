import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import warnings
warnings.simplefilter("ignore")


class FeatureExtraction:

    def __init__(self,x,y,test):
        self.x = x
        self.y = y
        self.test = test
        self.tokenizer = Tokenizer()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def Extraction(self):
        # Tokenize the words
        self.tokenizer.fit_on_texts(self.x)
        X_sequences = self.tokenizer.texts_to_sequences(self.x)
        # print(X_sequences)

        # Pad sequences to make them of the same length
        X_padded = pad_sequences(X_sequences)
        # print(X_padded)

        # Split the data into training and testing sets
        self.X_train,  self.X_test, self.y_train, self.y_test = train_test_split(X_padded, self.y, test_size=0.2,
                                                                                 random_state=42,shuffle=True)
        return self.X_train,  self.X_test, self.y_train, self.y_test, self.tokenizer.word_index

    def extraction_test(self,choose):
        # Tokenize the words
        test_sequences = self.tokenizer.texts_to_sequences(self.test)
        # print(len(test_sequences))

        # Pad sequences to make them of the same length
        test_padded = pad_sequences(test_sequences)
        # print(len(test_padded))

        if choose == 2:
            new_test_padded = []
            for i in range(0, 1000):
                temp = np.array(test_padded[i])
                zeros_to_pad = max(0, self.X_train.shape[1] - len(temp))
                pad_before = zeros_to_pad // 2
                pad_after = zeros_to_pad - pad_before
                temp = np.pad(temp, (pad_before, pad_after), mode='constant', constant_values=0)
                new_test_padded.append(temp)

            new_test_padded = np.array(new_test_padded)
            print(f"test pad1{new_test_padded.shape}")
        return test_padded
