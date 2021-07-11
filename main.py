from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
import numpy as np
import nltk
import string
import re
import matplotlib.pyplot as plt

import os
import shutil
import string
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__, static_folder="data")
Bootstrap(app)


@app.route('/', methods=["GET"])
def home():
    return render_template('home.html')


@app.route('/analisis', methods=["POST"])
def analisis():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)
    
    #LOAD DATA
    tweetfile = request.files['tweetfile']
    tweet_path = "./data/input_data.csv"
    tweetfile.save(tweet_path)
    test_data = pd.read_csv(tweet_path)
    test_data.drop(test_data.columns[[1, 2]], axis=1, inplace=True)
    
    #PREPROCESSING - REMOVE USERNAME
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt
    test_data['remove_user'] = np.vectorize(remove_pattern)(test_data['tweet'], "@[\w]*")
    
    #PREPROCESSING - REMOVE SYMBOLS, NUMBERS, WHITE SPACES, EMOJI, HASHTAG
    def remove(tweet):
        #remove angka
        tweet = re.sub('[0-9]+', '', tweet)
        # remove old style retweet text "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        # remove hashtags
        # only removing the hash # sign from the word
        tweet = re.sub(r'#', '', tweet)
        # remove URL Links
        tweet = re.sub(r"http\S+", "", tweet)
        # remove puntuation & emoji (remove all besides \w > word dan \s > space)
        tweet = re.sub(r"[^\w\s]", "", tweet)
        # remove new line
        tweet = re.sub(r"[\n]+", "", tweet)
        #remove leading and trailing spaces in a word
        tweet = re.sub(r"^\s+|\s+$", "", tweet)  # using OR sign to delete both
        #remove multiple space betwen words
        tweet = re.sub(r" +", " ", tweet)
        return tweet
    test_data['remove_char'] = test_data['remove_user'].apply(lambda x: remove(x))
    test_data.drop_duplicates(subset="remove_char", keep='first', inplace=True)

    #PREPROCESSING - STEMMING LOCAL KBBI
    dicts = {row[0]: row[1] for _, row in pd.read_csv("kbba.txt", delimiter="\t").iterrows()}
    def kbbi(text):
        token = text.split()
        final_string = ' '.join(str(dicts.get(word, word)) for word in token)
        return final_string
    test_data['tweet_clean1'] = test_data['remove_char'].apply(lambda x: kbbi(x))
    
    #PREPROCESSING - STEMMING SASTRAWI n STOPWORD NLTK
    #download stopwords NLTK
    nltk.download('stopwords')
    stopwords_indonesia = stopwords.words('indonesian')
    #import sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    #tokenize
    def clean_tweets(tweet):
        # tokenize tweets & lower-casing
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        tweet_tokens = tokenizer.tokenize(tweet)

        tweets_clean = []
        for word in tweet_tokens:
            if (word not in stopwords_indonesia):
                #tweets_clean.append(word)
                stem_word = stemmer.stem(word)  # stemming word
                tweets_clean.append(stem_word)
        return tweets_clean
    test_data['tweet_clean2'] = test_data['tweet_clean1'].apply(lambda x: clean_tweets(x))

    #PREPROCESSING - REMOVE SYMBOL 2
    def remove_punct(text):
        text = " ".join([char for char in text if char not in string.punctuation])
        return text
    test_data['Tweet'] = test_data['tweet_clean2'].apply(lambda x: remove_punct(x))

    #PREPROCESSING - DROP GARBAGE TWEETS
    #finalisation
    test_data.sort_values("Tweet", inplace=True)
    test_data.drop(test_data.columns[[0, 1, 2, 3, 4, 5]], axis=1, inplace=True)
    #delete rows which duplicating
    test_data.drop_duplicates(subset="Tweet", keep='first', inplace=True)
    #delete rows with certain keywords
    test_data = test_data[~test_data.Tweet.str.contains("hot sale")]
    test_data = test_data[~test_data.Tweet.str.contains("order")]
    test_data = test_data[~test_data.Tweet.str.contains("sleman")]
    test_data = test_data[~test_data.Tweet.str.contains("yogyakarta")]
    test_data = test_data[~test_data.Tweet.str.contains("diskon")]
    test_data = test_data[~test_data.Tweet.str.contains("disewakan")]
    test_data = test_data[~test_data.Tweet.str.contains("sewa")]
    test_data = test_data[~test_data.Tweet.str.contains("dijual")]
    test_data = test_data[~test_data.Tweet.str.contains("jual murah")]
    test_data = test_data[~test_data.Tweet.str.contains("jual cepat")]
    test_data = test_data[~test_data.Tweet.str.contains("promo ")]
    test_data = test_data[~test_data.Tweet.str.contains("mesan")]
    test_data = test_data[~test_data.Tweet.str.contains("ready")]
    test_data = test_data[~test_data.Tweet.str.contains("jual rumah")]
    test_data = test_data[~test_data.Tweet.str.contains("jual apartemen")]
    test_data = test_data[~test_data.Tweet.str.contains("jual ruko")]
    test_data = test_data[~test_data.Tweet.str.contains("jual tanah")]
    test_data = test_data[~test_data.Tweet.str.contains("jasa tukang")]
    test_data = test_data[~test_data.Tweet.str.contains("jasa bor")]
    test_data = test_data[~test_data.Tweet.str.contains("tukang taman")]
    test_data = test_data[~test_data.Tweet.str.contains("sedot wc")]
    test_data = test_data[~test_data.Tweet.str.contains("luas tanah")]
    test_data = test_data[~test_data.Tweet.str.contains("mayat")]
    test_data = test_data[~test_data.Tweet.str.contains("pkbday")]
    test_data = test_data[~test_data.Tweet.str.contains("jual sampah")]
    test_data = test_data[~test_data.Tweet.str.contains("murah")]
    test_data = test_data[~test_data.Tweet.str.contains("tukang kebun")]
    test_data = test_data[~test_data.Tweet.str.contains("malioboro")]
    test_data = test_data[~test_data.Tweet.str.contains("jasa")]

    #PREPARING FOR SENTIMENT ANALYSIS STEP
    test_data.dropna(axis=0, how='any', inplace=True)
    test_data['Num_words_text'] = test_data['Tweet'].apply(lambda x: len(str(x).split()))
    max_test_sentence_length = test_data['Num_words_text'].max()
    mask = test_data['Num_words_text'] > 2
    test_data = test_data[mask]

    #LOAD CNN MODEL
    new_model = tf.keras.models.load_model('static\model\model_analisis.h5')

    #TEST CNN MODEL TO test_data
    num_words = 20000
    tokenizer = Tokenizer(num_words=num_words, oov_token="unk")
    tokenizer.fit_on_texts(test_data['Tweet'].tolist())
    x_test = np.array(tokenizer.texts_to_sequences(
        test_data['Tweet'].tolist()))
    x_test = pad_sequences(x_test, padding='post', maxlen=50)

    #GENERATE PREDICTION
    predictions = new_model.predict(x_test)
    predict_results = predictions.argmax(axis=1)

    test_data['sentimen'] = predict_results
    test_data['sentimen'] = np.where(
        (test_data.sentimen == 0), 'negative', test_data.sentimen)
    test_data['sentimen'] = np.where(
        (test_data.sentimen == '1'), 'neutral', test_data.sentimen)
    test_data['sentimen'] = np.where(
        (test_data.sentimen == '2'), 'positive', test_data.sentimen)
    
    test_data.drop(test_data.columns[[1]], axis=1, inplace=True)

    labels = ['positive', 'negative', 'neutral']
    df = pd.DataFrame(test_data)
    df.to_csv('data/lingkungan_depok_result.csv', encoding='utf8', index=False)

    #VISUALISATION - PIE CHART
    fig, ax = plt.subplots(figsize=(3, 3))
    sizes = [count for count in df['sentimen'].value_counts()]
    labels = list(df['sentimen'].value_counts().index)
    explode = (0.1, 0, 0)
    ax.pie(x=sizes, labels=labels, autopct='%1.1f%%', explode=explode, textprops={'fontsize': 10})
    ax.set_title('Diagram Pie Perbandingan Sentimen', fontsize=10, pad=10)
    image_path = "./data/chart.png"
    plt.savefig(image_path)

    #RESULT DATA
    return render_template("analisis.html", column_names=df.columns.values, row_data=list(df.values.tolist()), zip=zip, chart=image_path)


if __name__ == '__main__':
    app.run(debug=True)
