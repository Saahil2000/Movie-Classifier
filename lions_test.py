import tensorflow as td
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
#Data is formed of ineteger encoded words - array of numbers
#print(train_data[0])
#Need to create mapping for numbers onto words - usually create your own

word_index = data.get_word_index()

#assign values starting at 1 + 3 so you can assiign your own values for 0 - 3 iin dictionary
word_index = {k:(v+3) for k, v in word_index.items()} # k and v = key and value
word_index["<PAD>"] = 0  #make each movie review the same length
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) #swap keys and values so integer points to word

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250) #Make each review 250 length
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

#Function to decode all of training and test data into human words
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded

model = keras.models.load_model("model.h5")

with open("four_lions.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")#want to remove all of these symbols as e.g. no word for art. or king,
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])

#prediction will give a score from 0-1 based on how poisitive/negative it is
