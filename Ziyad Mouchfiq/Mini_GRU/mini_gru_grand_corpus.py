import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

corpus = [
    "Artificial intelligence is changing the world",
    "Machine learning helps computers learn from data",
    "Deep learning is a part of artificial intelligence",
    "Artificial intelligence helps people solve problems",
    "Machine learning uses data to make predictions",
    "Deep learning uses neural networks",
    "Neural networks learn patterns from data",
    "Artificial intelligence is used in many industries",
    "Machine learning improves computer performance",
    "Deep learning can recognize images",
    "Artificial intelligence can understand human language",
    "Neural networks are important in deep learning",
    "Machine learning algorithms analyze data",
    "Artificial intelligence makes computers smarter",
    "Deep learning models require large amounts of data"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) + 1

sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        sequences.append(token_list[:i+1])

max_sequence_len = max(len(seq) for seq in sequences)

sequences = pad_sequences(
    sequences,
    maxlen=max_sequence_len,
    padding='pre'
)

X = sequences[:, :-1]
y = sequences[:, -1]

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

    output_word = tokenizer.index_word.get(predicted_word_index, "")

    if output_word == "":
        break

    seed_text += " " + output_word

print("Phrase générée :", seed_text)
print("Accuracy :", history.history['accuracy'][-1])
print("Loss :", history.history['loss'][-1])