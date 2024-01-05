import keras
from keras.layers import LSTM, BatchNormalization, Bidirectional, Dense, Dropout
from keras.layers import Input, Embedding, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
import warnings
warnings.simplefilter("ignore")
import pickle


class Models:

    def __init__(self,X_train,X_test,y_train,y_test,word_index):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.word_index = word_index

    def make_model_layers(self,choose):
        # Define the model
        model = None
        if int(choose) == 1:
            model = keras.Sequential()
            model.add(Embedding(input_dim=len(self.word_index) + 1, output_dim=128))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))
            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(Dropout(0.1))
            model.add(Bidirectional(LSTM(32)))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(3, activation='softmax'))
        elif int(choose) == 2:
            inputs = Input(shape=(self.X_train.shape[1],))
            x = Embedding(input_dim=len(self.word_index) + 1, output_dim=128)(inputs)
            print(x)

            # Transformer Block 1
            attn1 = MultiHeadAttention(num_heads=8, key_dim=128)(x, x)
            attn1 = Dropout(0.1)(attn1)
            # attn = Add()
            out1 = LayerNormalization(epsilon=1e-6)(x + attn1)

            # Transformer Block 2
            attn2 = MultiHeadAttention(num_heads=8, key_dim=128)(out1, out1)
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
        else:
            print("Undefined  model")

        # Compile the model
        # Gradient clipping involves forcing the gradient values (element-wise) to a specific minimum or maximum value
        # if the gradient exceeded an expected range.
        opt = keras.optimizers.Adam(learning_rate=0.0001, clipvalue=5.0)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        # opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)  # You can adjust the momentum value
        # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=2)

        # Train the model
        num_epochs = 5
        model.fit(self.X_train, self.y_train, epochs=num_epochs, validation_data=(self.X_test, self.y_test),
                  verbose=2,callbacks=[es])
        if int(choose) == 1:
            pickle.dump(model, open('NN_model.pk1', 'wb'))
        elif int(choose) == 2:
            pickle.dump(model, open('Transformer_model.pk1', 'wb'))