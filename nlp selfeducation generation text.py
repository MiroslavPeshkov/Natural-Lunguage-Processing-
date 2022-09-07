import os
import glob
import pandas as pd
import spacy
import copy
import nltk
import re
import time
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from string import punctuation
from tensorflow import keras
from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.utils import pad_sequences
from PyPDF2 import PdfReader
import warnings
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
import pydot
import graphviz
from sklearn.metrics.pairwise import euclidean_distances
from keras.preprocessing import text
from keras.preprocessing.sequence import skipgrams
from keras.layers import Dot
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Reshape
from keras.layers import Embedding
from nltk.tokenize import word_tokenize
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.utils import to_categorical
from random import randint
import re
from keras.preprocessing.text import Tokenizer
import warnings

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("ru_core_news_sm")
remove_terms = punctuation + '0123456789'
all_stopwords = nlp.Defaults.stop_words
warnings.filterwarnings("ignore")

count_char_corpus = 700000


def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^а-яА-Я]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[а-яА-Я]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence.lower()


all_files = glob.glob('Files/*.pdf')
corpus = ""
count = 0
# add in corpus rows of text from PDF files
for file in all_files:
    count += 1
    reader = PdfReader(file)
    for page in reader.pages:
        corpus += page.extract_text() + "\n"

# Devide corpus in 700000 characters
corpus = corpus[:count_char_corpus]

# implement function of preprocessing text
corpus = preprocess_text(corpus)

# Devide corpus in tokens
text_words = (word_tokenize(corpus))
n_words = len(text_words)
# remove no unique words
unique_words = len(set(text_words))

print('Total Words: %d' % n_words)
print('Unique Words: %d' % unique_words)

# create vectorize a text corpus by Keras
tokenizer = Tokenizer(num_words=unique_words)
tokenizer.fit_on_texts(text_words)
# vocabular size for next use in text generation model
vocab_size = len(tokenizer.word_index) + 1
# index of words
word_2_index = tokenizer.word_index

print(text_words[10])
print(word_2_index[text_words[10]])


input_sequence = []
output_words = []
# input length of sequenses
input_seq_length = 1000

for i in range(0, n_words - input_seq_length, 1):
    in_seq = text_words[i:i + input_seq_length]
    out_seq = text_words[i + input_seq_length]
    input_sequence.append([word_2_index[word] for word in in_seq])
    output_words.append(word_2_index[out_seq])

print(input_sequence[0])

# create X matrix
X = np.reshape(input_sequence, (len(input_sequence), input_seq_length, 1))
X = X / float(vocab_size)

# Converts a class vector (integers) to binary class matrix
y = to_categorical(output_words)

# create generation text model
model = Sequential()
model.add(LSTM(800, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(800, return_sequences=True))
model.add(LSTM(800))
model.add(Dense(y.shape[1], activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam')

# NOTE! My model train more than one week. It depends from size of epoch and characters in input text
model.fit(X, y, batch_size=64, epochs=5, verbose=1)

# Save model in current directory
PATH = os.getcwd()
model.save(PATH)

# Generate text from model
random_seq_index = np.random.randint(0, len(input_sequence) - 1)
random_seq = input_sequence[random_seq_index]

index_2_word = dict(map(reversed, word_2_index.items()))

word_sequence = [index_2_word[value] for value in random_seq]

for i in range(1000):
    int_sample = np.reshape(random_seq, (1, len(random_seq), 1))
    int_sample = int_sample / float(vocab_size)

    # model try to predict sequence word and generate text
    predicted_word_index = model.predict(int_sample, verbose=0)

    predicted_word_id = np.argmax(predicted_word_index)
    seq_in = [index_2_word[index] for index in random_seq]

    word_sequence.append(index_2_word[predicted_word_id])

    random_seq.append(predicted_word_id)
    random_seq = random_seq[1:len(random_seq)]

final_output = ""
for word in word_sequence:
    final_output = final_output + " " + word

with open('Generation text_1', 'w') as f:
    f.write(final_output)
print(final_output)
