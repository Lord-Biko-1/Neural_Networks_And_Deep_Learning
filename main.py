# pip install -U deep-translator
# import nltk
# nltk.download('stopwords')
# pip install emoji

import pandas as pd
import preprocess
import feature_extraction
import Models
import Edit_the_prediction
import warnings
import pickle
warnings.simplefilter("ignore")

# Read data
data = pd.read_csv("train.CSV")
y = data["rating"]
for i in range(len(y)):
    if y[i] == -1:
        y[i] = 0
    elif y[i] == 0:
        y[i] = 1
    if y[i] == 1:
        y[i] = 2
# print(y)
X = data["review_description"]

pre = preprocess.Preprocess(X)
X = pre.preprocess_text()
# print(X[44])
# print(X[54])
# print(X)

# Test the model
data_test = pd.read_csv("test _no_label.csv")
test = data_test["review_description"]
pre = preprocess.Preprocess(test)
test = pre.preprocess_text()

# feature extraction
extract = feature_extraction.FeatureExtraction(X,y,test)
X_train, X_test, y_train, y_test, word_index = extract.Extraction()

# Define the models
print("1) Train new model\n2) Load trained model\n")
train_choice = input()
if(train_choice == "1"):
    print("choose your model 1:LSTM , 2: Transformer \n")
    choose = input()

    model = Models.Models(X_train, X_test, y_train, y_test, word_index)
    model.make_model_layers(choose)
else:
    print("1) LSTM\n2) Transformer\n")
    model_choice = input()

    loaded_model = None
    # load the model
    if int(model_choice) == 1:
        loaded_model = pickle.load(open('NN_model.pk1', 'rb'))
    elif int(model_choice) == 2:
        loaded_model = pickle.load(open('Transformer_model.pk1', 'rb'))
        # Pad sequences of test to make them of the same length
    test_padded = extract.extraction_test(int(model_choice))
    # print(len(test_padded))

    predict = loaded_model.predict(test_padded)
    # print(len(predict))
    edit = Edit_the_prediction.EditPredict(predict)
    final = edit.Edit()

    output = pd.DataFrame({'ID': data_test.ID, 'rating': final})
    if int(model_choice) == 1:
        output.to_csv('submission.csv', index=False)
    elif int(model_choice) == 2:
        output.to_csv('submission_Transformer.csv', index=False)
    print("Your submission was successfully saved!")
