import numpy as np
import pickle
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam


# Constants
SEQUENCE_LEN = 50
EMBEDDING_DIM = 256
LSTM_UNITS = 256

MAX_VOCAB_SIZE = 10000  # or whatever size you think is appropriate
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')


def create_model(vocab_size=MAX_VOCAB_SIZE):
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=SEQUENCE_LEN-1))
    model.add(LSTM(LSTM_UNITS))
    model.add(Dense(vocab_size, activation='softmax'))
       # Define your custom learning rate
    custom_adam_optimizer = Adam(learning_rate=0.01)  # for example, 0.0005
    
    model.compile(optimizer=custom_adam_optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def _get_training_callbacks():
    checkpoint = ModelCheckpoint("final_base_model.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    return [checkpoint, early_stopping, reduce_lr]

def _get_training_callbacks_local(model_name):
    local_model_name = f"{model_name}_local_model.h5"
    model_path = os.path.join("user_local_models", local_model_name)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    return [checkpoint, early_stopping, reduce_lr]

def _get_training_callbacks_global(model_name):
    local_model_name = f"{model_name}_global_model.h5"
    model_path = os.path.join("user_global_models", local_model_name)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    return [checkpoint, early_stopping, reduce_lr]

def train_model(text, model=None):
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in text.split('.'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    input_sequences = pad_sequences(
        input_sequences, maxlen=SEQUENCE_LEN, padding='pre')
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    # One-hot encode the labels
    y = to_categorical(y, num_classes=MAX_VOCAB_SIZE)

    callbacks = _get_training_callbacks()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=12)

    if model is None:
        model = create_model()

    model.fit(X_train, y_train, epochs=75, verbose=1, batch_size=150, validation_data=(X_test, y_test), callbacks=callbacks)
    return model, X_test, y_test


def predict_next_word(model, text, tokenizerr):
    print("in predict")
    print("Original text:", text) 
    token_list = tokenizerr.texts_to_sequences([text])[0]
    print(token_list)
    token_list = pad_sequences(
        [token_list], maxlen=SEQUENCE_LEN-1, padding='pre')
    print("hello")
    print(token_list)
    prediction = model.predict(token_list)
    predicted_word = tokenizerr.index_word[np.argmax(prediction)]
    return predicted_word


def save_tokenizer(tokenizer, text):
    tokenizer.fit_on_texts([text])
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def predict_top_n_words(model, tokenizerr, text, n=3):
    token_list = tokenizerr.texts_to_sequences([text])[0]
    token_list = pad_sequences(
        [token_list], maxlen=SEQUENCE_LEN-1, padding='pre')
    prediction = model.predict(token_list)

    # Get the top n word indices
    top_indices = prediction[0].argsort()[-n:][::-1]
    predicted_words = [tokenizerr.index_word[idx] for idx in top_indices]

    return predicted_words

def predict_top_n_local_global_words(model1, model2, tokenizer, text, n1=3, n2=2):
    # Prepare the token list
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=SEQUENCE_LEN-1, padding='pre')

    # Get predictions from both models
    prediction1 = model1.predict(token_list)
    prediction2 = model2.predict(token_list)

    # Get the top n word indices from both predictions
    top_indices1 = prediction1[0].argsort()[-n1:][::-1]
    top_indices2 = prediction2[0].argsort()[-n2:][::-1]

    # Retrieve the corresponding words
    predicted_words1 = [tokenizer.index_word[idx] for idx in top_indices1]
    predicted_words2 = [tokenizer.index_word[idx] for idx in top_indices2]

    # Combine the predictions, with those from the first model coming first
    combined_predictions = predicted_words1 + predicted_words2

    return combined_predictions


def predict_top_n_words_local_global_unique_words(model1, model2, tokenizer, text, n1=3, n2=2):
    # Prepare the token list
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=SEQUENCE_LEN-1, padding='pre')

    # Predict the next words with both models
    prediction1 = model1.predict(token_list)
    prediction2 = model2.predict(token_list)

    # Get the top n word indices from the first prediction
    top_indices1 = prediction1[0].argsort()[-n1:][::-1]
    # Translate indices to words
    predicted_words1 = [tokenizer.index_word[idx] for idx in top_indices1]

    # For the second model, we will retrieve more than 'n2' top indices in case of overlaps
    # This will give us a pool of words to choose from if there are duplicates
    extended_top_indices2 = prediction2[0].argsort()[-(n1+n2):][::-1]  # increase the pool size

    predicted_words2 = []
    for idx in extended_top_indices2:
        word = tokenizer.index_word[idx]
        if word not in predicted_words1 and len(predicted_words2) < n2:
            predicted_words2.append(word)
        # Break the loop if we've found enough unique words
        if len(predicted_words2) == n2:
            break

    # If, after checking the extended list, we still don't have enough unique words,
    # you may want to handle this scenario (e.g., by notifying the user or using placeholders).
    # This scenario is unlikely but possible if the models are very similar or the vocabulary is limited.

    # Combine the predictions
    combined_predictions = predicted_words1 + predicted_words2

    return combined_predictions


def create_sequences(text, tokenizer):
    input_sequences = []
    for line in text.split('.'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

def prepare_data(sequences, tokenizer):
    # Pad sequences for consistent length
    input_sequences = pad_sequences(sequences, maxlen=SEQUENCE_LEN, padding='pre')

    # Separate predictors and target
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    
    # One-hot encode the target variable
    y = to_categorical(y, num_classes=MAX_VOCAB_SIZE)

    return X, y


def train_local_model(training_text, validation_text, tokenizer, model_name, model=None, learning_rate=0.005):
    # Update the tokenizer based on the training text
    tokenizer.fit_on_texts([training_text])
    
    # Create sequences for the training text
    train_sequences = create_sequences(training_text, tokenizer)
    
    # Prepare the training data
    X_train, y_train = prepare_data(train_sequences, tokenizer)

    # Create sequences for the validation text
    validation_sequences = create_sequences(validation_text, tokenizer)
    
    # Prepare the validation data
    X_val, y_val = prepare_data(validation_sequences, tokenizer)

    # Define your callbacks, optimizer, and compile the model as before
    callbacks = _get_training_callbacks_local(model_name)
    new_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=new_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=30, verbose=1, batch_size=50, validation_data=(X_val, y_val), callbacks=callbacks)
   
    save_trained_tokenizer(tokenizer,model_name)
    return model, tokenizer, X_val, y_val


def save_trained_tokenizer(tokenizer, model_name):
    user_tokenizer_name = f"{model_name}_tokenizer.pickle"
    tokenizer_path = os.path.join("user_tokenizers", user_tokenizer_name)
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
