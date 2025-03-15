import numpy as np
import pickle
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Parameters
vocab_size = 10000  # Number of words to use from the IMDB dataset
max_length = 100    # Maximum length of sequences

# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Padding sequences to ensure uniform input size
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

# Model architecture
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)

# Save the model
model.save('sentiment_model.h5')

# Save the tokenizer for later use
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(imdb.get_word_index().keys())
with open('tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved successfully.")
