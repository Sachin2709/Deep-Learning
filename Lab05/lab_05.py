# ==============================
# PART 1: LSTM TEXT CLASSIFICATION (IMDB)
# ==============================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Step 1: Load Dataset
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Step 2: Padding
max_len = 200
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# Step 3: Build Model
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_len))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

# Step 4: Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 5: Train
history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2)

# Step 6: Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("LSTM Test Accuracy:", acc)

# ==============================
# 📊 GRAPHS (FIXED - BOTH ACCURACY + LOSS)
# ==============================

# Accuracy Graph
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# Loss Graph (NEW FIX)
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()


# ==============================
# 🔹 PART 2: SEQ2SEQ MODEL (ENCODER-DECODER LSTM)
# ==============================

# Simple dataset
input_texts = ["hi", "hello", "bye", "thanks"]
target_texts = ["salut", "bonjour", "au revoir", "merci"]

# Tokenization
input_tokenizer = Tokenizer()
target_tokenizer = Tokenizer()

input_tokenizer.fit_on_texts(input_texts)
target_tokenizer.fit_on_texts(target_texts)

input_seq = input_tokenizer.texts_to_sequences(input_texts)
target_seq = target_tokenizer.texts_to_sequences(target_texts)

# Padding
max_input_len = max(len(seq) for seq in input_seq)
max_target_len = max(len(seq) for seq in target_seq)

encoder_input_data = pad_sequences(input_seq, maxlen=max_input_len, padding='post')
decoder_input_data = pad_sequences(target_seq, maxlen=max_target_len, padding='post')

# One-hot encoding
decoder_output_data = np.zeros(
    (len(target_texts), max_target_len, len(target_tokenizer.word_index) + 1),
    dtype="float32"
)

for i, seq in enumerate(target_seq):
    for t, word in enumerate(seq):
        decoder_output_data[i, t, word] = 1.0

# Model Parameters
latent_dim = 64

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(len(input_tokenizer.word_index)+1, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(len(target_tokenizer.word_index)+1, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

decoder_dense = Dense(len(target_tokenizer.word_index)+1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Seq2Seq Model
seq2seq_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile
seq2seq_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
seq2seq_model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_output_data,
    epochs=200,
    verbose=1
)

print("Seq2Seq Model Trained Successfully!")