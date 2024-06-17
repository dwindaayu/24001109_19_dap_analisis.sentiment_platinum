import pickle
import clean_helper as c

KAMUS_ALAY = c.kamus_alay()

# Load tfidf vectorizer
file_tfidf = open("tfidf_vectorizer.p", 'rb')
TFIDF_VECTORIZER = pickle.load(file_tfidf)
file_tfidf.close()

# Load MLP Classifier Model
file_mlp = open('model_mlp.p','rb')
MODEL_MLP = pickle.load(file_mlp)
file_mlp.close()

# Function for predict by Neural Network MLP Classifier
def prediction_by_mlp(str):
    # data cleansing
    str = c.clean(str)
    str = c.word_substitute(str, KAMUS_ALAY)
    # feature extraction
    text = TFIDF_VECTORIZER.transform([str])
    #predict
    result = MODEL_MLP.predict(text)
    print(result)
    if result[0] == 0:
        return 'neutral'
    elif result[0] == 1:
        return 'positive'
    elif result[0] == 2:
        return 'negative'
    else:
        return 'neutral'

    # if result[0] == 1:
    #     return 'negative'
    # elif result[1] == 1:
    #     return 'neutral'
    # elif result[2] == 1:
    #     return 'positive'
    # else:
    #     return 'neutral'