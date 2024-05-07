#Adapted from Colab Notebook

import pandas as pd
import regex as re
import math
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv("/content/drive/MyDrive/data3.csv")
X = list(df["X"])[:150]
y = np.array(df["y"])[:150]

df2 = pd.read_csv("/content/drive/MyDrive/data3.csv")
df2["X2"] = df["X"]
X = df2[["X", "X2"]][:150]
y = np.array(df2["y"])

from statistics import mean
y_test = [mean(y)] * len(y)
mean_squared_error(y, y_test)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(X)
padded_sequences = pad_sequences(sequences)


embedding_dim = 100  # Adjust based on the GloVe embedding you choose
vocab_size = len(word_index) + 1

model = tf.keras.Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=padded_sequences.shape[1], trainable=False),
    Bidirectional(LSTM(32, return_sequences=True)),
    Bidirectional(LSTM(16)),
    Dense(8, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Output layer for regression
])

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(padded_sequences, y, epochs=20, validation_split=0.1)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error

X_test = list(df["X"][150:])
y_test = np.array(df["y"])[150:]
print(len(X_test))
print(len(y_test))

tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(X_test)
sequences2 = tokenizer.texts_to_sequences(X_test)
padded_sequences2 = pad_sequences(sequences2)
predictions = model.predict(padded_sequences2)

print(len(predictions))

mse = mean_squared_error(y_test, predictions)
print(mse)