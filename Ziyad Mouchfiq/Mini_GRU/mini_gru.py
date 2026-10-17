from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.layers import LSTM
import time
from tensorflow.keras.layers import Dropout

corpus = [
    "Artificial intelligence is changing the world",
    "Machine learning helps computers learn from data"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

print(tokenizer.word_index)

sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        sequences.append(n_gram_sequence)

print(sequences)



max_sequence_len = max(len(seq) for seq in sequences)

sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')

print(sequences)



X = sequences[:, :-1]
y = sequences[:, -1]

print("X =", X)
print("y =", y)



total_words = len(tokenizer.word_index) + 1

model = Sequential([
    Embedding(input_dim=total_words, output_dim=10),
    GRU(100),
    Dense(total_words, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.build(input_shape=(None, max_sequence_len - 1))

model.summary()

history = model.fit(
    X,
    y,
    epochs=500,
    verbose=1
)

seed_text = "Artificial intelligence"
next_words = 3

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding='pre'
    )

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=-1)[0]

    output_word = ""

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            output_word = word
            break

    if output_word == "":
        break

    seed_text += " " + output_word

print("Phrase générée :", seed_text)



lstm_model = Sequential([
    Embedding(input_dim=total_words, output_dim=10),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

lstm_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

start_time = time.time()

history_lstm = lstm_model.fit(
    X,
    y,
    epochs=500,
    verbose=0
)

lstm_time = time.time() - start_time

print("Accuracy LSTM :", history_lstm.history['accuracy'][-1])
print("Loss LSTM :", history_lstm.history['loss'][-1])
print("Temps LSTM :", round(lstm_time, 2), "secondes")

print("Accuracy GRU :", history.history['accuracy'][-1])
print("Loss GRU :", history.history['loss'][-1])



dropout_model = Sequential([
    Embedding(input_dim=total_words, output_dim=10),
    GRU(100),
    Dropout(0.2),
    Dense(total_words, activation='softmax')
])

dropout_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history_dropout = dropout_model.fit(
    X,
    y,
    epochs=500,
    verbose=0
)

print("Accuracy GRU :", history.history['accuracy'][-1])
print("Loss GRU :", history.history['loss'][-1])

print("Accuracy GRU + Dropout :", history_dropout.history['accuracy'][-1])
print("Loss GRU + Dropout :", history_dropout.history['loss'][-1])