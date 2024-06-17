import pandas as pd
import clean_helper as c
import prediction_helper as pr
import sqlite3
import pickle, re
# import keras
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
# from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences




DB_FILE = 'sentiment_analysis.db'

app = Flask(__name__)

app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': 'API Documentation for Sentiment Analysis',
    'version': '1.0.0',
    'description': 'Dokumentasi API untuk Sentiment Analysis',
    }
    # host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template,             
                  config=swagger_config)

max_features = 100000
max_len = 91
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

sentiment = {0: 'neutral', 1: 'positive', 2: 'negative'}

def cleansing(sent):
    string = sent.lower()
    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
    return string

file = open('x_pad_sequence.pickle','rb')
file_pad_sequences = pickle.load(file)
file.close()

model_lstm = tf.keras.models.load_model('my_lstm_model.h5')


@app.route('/', methods=['GET'])
def hello_world():
    return 'API for Sentiment Analysis'

@swag_from("sentiment_text_nnmlp.yml", methods=['POST'])
@app.route('/sentiment-text-nnmlp', methods=['POST'])
def sentiment_text_nnmlp():
    # required validation
    if 'text' not in request.form:
        return res('text is required', 400)
    
    text = request.form['text']

    # predict
    hasil = pr.prediction_by_mlp(text) #
    # save db
    data = [(text, hasil, "mlp_classifier")]
    insert_into_texts(data)
    
    return res({
        "input_text": text,
        "sentiment": hasil,
        "model_type": "mlp_classifier"
    })

@swag_from("sentiment_file_nnmlp.yml", methods=['POST'])
@app.route('/sentiment-file-nnmlp', methods=['POST'])
def sentiment_file_nnmlp():
    # required validation
    if 'file' not in request.files:
        return res('file is required', 400)
    
    file = request.files.getlist('file')[0]
    df = pd.read_csv(file)
    texts = df.text.to_list()

    res_arr = []
    data_insert = []
    for text in texts:
        # predict
        hasil = pr.prediction_by_mlp(text) 
        # append array insert
        data_insert.append((text, hasil, "mlp_classifier"))

        res_arr.append({
        "input_text": text,
        "sentiment": hasil,
        "model_type": "mlp_classifier"
    })
    
    # save db
    insert_into_texts(data_insert)
    return res(res_arr)

@swag_from("sentiment_text_lstm.yml", methods=['POST'])
@app.route('/sentiment-text-lstm', methods=['POST'])
def sentiment_text_lstm():
    # required validation
    if 'text' not in request.form:
        return res('text is required', 400)
    
    text = request.form['text']

    # predict
    # PREDISKI LSTM KODING DISIN
    cleaned_text = cleansing(text)
    
    predicted = tokenizer.texts_to_sequences([cleaned_text])

    # sequences = tokenizer.texts_to_sequences([cleaned_text])
    # padded_sequences = pad_sequences(sequences, maxlen=file_pad_sequences.shape[1])

    # predicted = tokenizer_lstm.texts_to_sequences(text)
    # print("predicted", predicted)
    # guess = tf.keras.preprocessing.sequence.pad_sequences(predicted, maxlen=82)
    # guess = tf.keras.preprocessing.sequence.pad_sequences(padded_sequences)
    # guess = model_lstm.predict(padded_sequences)
    guess = pad_sequences(predicted, maxlen=91)
    prediction = model_lstm.predict(guess)
    print(type([str]))
    print("raw", prediction)
    print("prediction", np.argmax(prediction[0]))
    hasil = sentiment[np.argmax(prediction[0])]

    # save db
    data = [(text, hasil, "lstm")]
    insert_into_texts(data)
    
    return res({
        "input_text": text,
        "sentiment": hasil,
        "model_type": "lstm"
    })

@swag_from("sentiment_file_lstm.yml", methods=['POST'])
@app.route('/sentiment-file-lstm', methods=['POST'])
def sentiment_file_lstm():
    # required validation
    if 'file' not in request.files:
        return res('file is required', 400)
    
    file = request.files.getlist('file')[0]
    df = pd.read_csv(file)
    texts = df.text.to_list()

    res_arr = []
    data_insert = []
    for text in texts:
        # predict
        # predict
        cleaned_text = cleansing(text)
    
        predicted = tokenizer.texts_to_sequences([cleaned_text])

    # predicted = tokenizer_lstm.texts_to_sequences(text)
    # print("predicted", predicted)
    # guess = tf.keras.preprocessing.sequence.pad_sequences(predicted, maxlen=82)
        guess = pad_sequences(predicted, maxlen=91)
        prediction = model_lstm.predict(guess)
        print(type([str]))
        print("raw", prediction)
        print("prediction", np.argmax(prediction[0]))
        hasil = sentiment[np.argmax(prediction[0])]
        # append array insert
        data_insert.append((text, hasil, "lstm"))
        hasil = sentiment[np.argmax(guess)]

        res_arr.append({
            "input_text": text,
            "sentiment": hasil,
            "model_type": "lstm"
    })
    
    # save db
    insert_into_texts(data_insert)
    return res(res_arr)


def res(data, code = 200):
    return jsonify({
        "status_code": code,
        "data": data
    }), code

def insert_into_texts(data):
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.cursor().executemany("INSERT INTO texts (input_text, sentiment, model_type) VALUES (?, ?, ?)", data)
        conn.commit()
        print ("success insert to texts")
    except sqlite3.Error as e:
        conn.rollback()
        print ("failed insert to texts", str(e))
    conn.cursor().close()
    conn.close()

if __name__ == '__main__':
    app.run()