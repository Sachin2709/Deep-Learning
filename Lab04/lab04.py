import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Preprocessing
max_len = 200
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# Model builder
def build_model(model_type):
    model = Sequential()
    
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len))
    
    if model_type == "RNN":
        model.add(SimpleRNN(32))
    elif model_type == "LSTM":
        model.add(LSTM(32))
    elif model_type == "GRU":
        model.add(GRU(32))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Train models
models = ["RNN", "LSTM", "GRU"]

for m in models:
    print(f"\nTraining {m} model")
    model = build_model(m)
    model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
    
    loss, acc = model.evaluate(X_test, y_test)
    print(f"{m} Test Accuracy:", acc)