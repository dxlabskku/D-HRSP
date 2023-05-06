from preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from tqdm import tqdm_notebook
from keras.layers import LSTM, Activation, Dropout, Dense, Input, Bidirectional, Embedding
from keras.models import Model, Sequential, load_model
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, plot_roc_curve, roc_curve, f1_score, precision_score, recall_score, confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D, GlobalMaxPooling1D
from keras.layers import MaxPool1D
from keras.models import Model

def lstm_mecab_target(data):
    X = data['clean_words_mecab']
    Y = data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42, stratify = Y_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_tr)
    vocab_size = len(tokenizer.word_index) + 1
    #print(vocab_size)
    data_train = tokenizer.texts_to_sequences(X_tr)
    data_valid = tokenizer.texts_to_sequences(X_val)
    data_test = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(i) for i in data_train)

    X_train = pad_sequences(data_train, maxlen = max_len)
    X_valid = pad_sequences(data_valid, maxlen = max_len)
    X_test = pad_sequences(data_test, maxlen = max_len)

    vocab_size = len(tokenizer.word_index)
    embedding_dim = 100
    hidden_units = 128

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Bidirectional(LSTM(hidden_units)))
    model.add(Activation('ReLU'))
    #model.add(Bidirectional(LSTM(32, return_sequences = True)))
    #model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1, activation = 'sigmoid'))

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
    mc = ModelCheckpoint('Models/lstm_mecab_binary_best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
    history = model.fit(X_train, Y_tr, epochs = 15, callbacks = [es, mc], batch_size = 16, validation_split = 0.2)

    y_pred_train = model.predict(X_train)
    score_train= [i for j in y_pred_train.tolist() for i in j]
    y_pred_train = []
    for i in score_train:
        if i > 0.5 :
            y_pred_train.append(1)
        else:
            y_pred_train.append(0)

    y_pred_valid = model.predict(X_valid)
    score_valid = [i for j in y_pred_valid.tolist() for i in j]
    y_pred_valid = []
    for i in score_valid:
        if i > 0.5 :
            y_pred_valid.append(1)
        else:
            y_pred_valid.append(0)

    y_pred_test = model.predict(X_test)
    score_test = [i for j in y_pred_test.tolist() for i in j]
    y_pred_test= []
    for i in score_test:
        if i > 0.5 :
            y_pred_test.append(1)
        else:
            y_pred_test.append(0)   

    report = classification_report(Y_tr, y_pred_train, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (10, 10))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    report = classification_report(Y_val, y_pred_valid, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)
    
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    report = classification_report(Y_test, y_pred_test, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])
    
    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_test))

def lstm_mecab_target_sampled(data):
    data_0 = data[data['target'] == 0]
    data_1 = data[data['target'] == 1]
    data_1 = data_1.sample(n = 2590, replace = False)
    data = pd.concat([data_0,data_1], axis = 0)

    X = data['clean_words_mecab']
    Y = data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42, stratify = Y_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_tr)
    vocab_size = len(tokenizer.word_index) + 1
    #print(vocab_size)
    data_train = tokenizer.texts_to_sequences(X_tr)
    data_valid = tokenizer.texts_to_sequences(X_val)
    data_test = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(i) for i in data_train)

    X_train = pad_sequences(data_train, maxlen = max_len)
    X_valid = pad_sequences(data_valid, maxlen = max_len)
    X_test = pad_sequences(data_test, maxlen = max_len)

    vocab_size = len(tokenizer.word_index)
    embedding_dim = 100
    hidden_units = 128

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Bidirectional(LSTM(hidden_units)))
    model.add(Activation('ReLU'))
    #model.add(Bidirectional(LSTM(32, return_sequences = True)))
    #model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1, activation = 'sigmoid'))

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
    mc = ModelCheckpoint('Models/lstm_mecab_binary_best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
    history = model.fit(X_train, Y_tr, epochs = 15, callbacks = [es, mc], batch_size = 16, validation_split = 0.2)

    y_pred_train = model.predict(X_train)
    score_train= [i for j in y_pred_train.tolist() for i in j]
    y_pred_train = []
    for i in score_train:
        if i > 0.5 :
            y_pred_train.append(1)
        else:
            y_pred_train.append(0)

    y_pred_valid = model.predict(X_valid)
    score_valid = [i for j in y_pred_valid.tolist() for i in j]
    y_pred_valid = []
    for i in score_valid:
        if i > 0.5 :
            y_pred_valid.append(1)
        else:
            y_pred_valid.append(0)

    y_pred_test = model.predict(X_test)
    score_test = [i for j in y_pred_test.tolist() for i in j]
    y_pred_test= []
    for i in score_test:
        if i > 0.5 :
            y_pred_test.append(1)
        else:
            y_pred_test.append(0)   

    report = classification_report(Y_tr, y_pred_train, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (10, 10))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    report = classification_report(Y_val, y_pred_valid, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)
    
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    report = classification_report(Y_test, y_pred_test, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])
    
    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_test))

def cnn_mecab_target(data):
    X = data['clean_words_mecab']
    Y = data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42, stratify = Y_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_tr)
    vocab_size = len(tokenizer.word_index) + 1
    data_train = tokenizer.texts_to_sequences(X_tr)
    data_valid = tokenizer.texts_to_sequences(X_val)
    data_test = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(i) for i in data_train)

    X_train = pad_sequences(data_train, maxlen = max_len)
    X_valid = pad_sequences(data_valid, maxlen = max_len)
    X_test = pad_sequences(data_test, maxlen = max_len)

    vocab_size = len(tokenizer.word_index)
    embedding_dim = 100

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(layers.Conv1D(128, 5, activation = 'relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(32, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
    mc = ModelCheckpoint('Models/cnn_mecab_binary_best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

    history = model.fit(X_train, Y_tr, epochs = 15, callbacks = [es, mc], batch_size = 16, validation_split = 0.2)

    y_pred_train = model.predict(X_train)
    score_train= [i for j in y_pred_train.tolist() for i in j]
    y_pred_train = []
    for i in score_train:
        if i > 0.5 :
            y_pred_train.append(1)
        else:
            y_pred_train.append(0)

    y_pred_valid = model.predict(X_valid)
    score_valid = [i for j in y_pred_valid.tolist() for i in j]
    y_pred_valid = []
    for i in score_valid:
        if i > 0.5 :
            y_pred_valid.append(1)
        else:
            y_pred_valid.append(0)

    y_pred_test = model.predict(X_test)
    score_test = [i for j in y_pred_test.tolist() for i in j]
    y_pred_test= []
    for i in score_test:
        if i > 0.5 :
            y_pred_test.append(1)
        else:
            y_pred_test.append(0)   

    report = classification_report(Y_tr, y_pred_train, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (10, 10))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    report = classification_report(Y_val, y_pred_valid, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)
    
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    report = classification_report(Y_test, y_pred_test, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])
    
    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_test))

def cnn_mecab_target_sampled(data):
    data_0 = data[data['target'] == 0]
    data_1 = data[data['target'] == 1]
    data_1 = data_1.sample(n = 2590, replace = False)
    data = pd.concat([data_0,data_1], axis = 0)

    X = data['clean_words_mecab']
    Y = data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42, stratify = Y_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_tr)
    vocab_size = len(tokenizer.word_index) + 1
    data_train = tokenizer.texts_to_sequences(X_tr)
    data_valid = tokenizer.texts_to_sequences(X_val)
    data_test = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(i) for i in data_train)

    X_train = pad_sequences(data_train, maxlen = max_len)
    X_valid = pad_sequences(data_valid, maxlen = max_len)
    X_test = pad_sequences(data_test, maxlen = max_len)

    vocab_size = len(tokenizer.word_index)
    embedding_dim = 100

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(layers.Conv1D(128, 5, activation = 'relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(32, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
    mc = ModelCheckpoint('Models/cnn_mecab_binary_best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

    history = model.fit(X_train, Y_tr, epochs = 15, callbacks = [es, mc], batch_size = 16, validation_split = 0.2)

    y_pred_train = model.predict(X_train)
    score_train= [i for j in y_pred_train.tolist() for i in j]
    y_pred_train = []
    for i in score_train:
        if i > 0.5 :
            y_pred_train.append(1)
        else:
            y_pred_train.append(0)

    y_pred_valid = model.predict(X_valid)
    score_valid = [i for j in y_pred_valid.tolist() for i in j]
    y_pred_valid = []
    for i in score_valid:
        if i > 0.5 :
            y_pred_valid.append(1)
        else:
            y_pred_valid.append(0)

    y_pred_test = model.predict(X_test)
    score_test = [i for j in y_pred_test.tolist() for i in j]
    y_pred_test= []
    for i in score_test:
        if i > 0.5 :
            y_pred_test.append(1)
        else:
            y_pred_test.append(0)   

    report = classification_report(Y_tr, y_pred_train, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (10, 10))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    report = classification_report(Y_val, y_pred_valid, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)
    
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    report = classification_report(Y_test, y_pred_test, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])
    
    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_test))

def lstm_khaiii_target(data):
    X = data['clean_words_khaiii']
    Y = data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42, stratify = Y_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_tr)
    vocab_size = len(tokenizer.word_index) + 1
    data_train = tokenizer.texts_to_sequences(X_tr)
    data_valid = tokenizer.texts_to_sequences(X_val)
    data_test = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(i) for i in data_train)

    X_train = pad_sequences(data_train, maxlen = max_len)
    X_valid = pad_sequences(data_valid, maxlen = max_len)
    X_test = pad_sequences(data_test, maxlen = max_len)

    vocab_size = len(tokenizer.word_index)
    embedding_dim = 100
    hidden_units = 64

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Bidirectional(LSTM(hidden_units)))
    model.add(Activation('ReLU'))
    model.add(Dense(1, activation = 'sigmoid'))

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 4)
    mc = ModelCheckpoint('Models/lstm_khaiii_binary_best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
    history = model.fit(X_train, Y_tr, epochs = 15, callbacks = [es, mc], batch_size = 32, validation_split = 0.2)

    y_pred_train = model.predict(X_train)
    score_train= [i for j in y_pred_train.tolist() for i in j]
    y_pred_train = []
    for i in score_train:
        if i > 0.5 :
            y_pred_train.append(1)
        else:
            y_pred_train.append(0)

    y_pred_valid = model.predict(X_valid)
    score_valid = [i for j in y_pred_valid.tolist() for i in j]
    y_pred_valid = []
    for i in score_valid:
        if i > 0.5 :
            y_pred_valid.append(1)
        else:
            y_pred_valid.append(0)

    y_pred_test = model.predict(X_test)
    score_test = [i for j in y_pred_test.tolist() for i in j]
    y_pred_test= []
    for i in score_test:
        if i > 0.5 :
            y_pred_test.append(1)
        else:
            y_pred_test.append(0)   

    report = classification_report(Y_tr, y_pred_train, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (10, 10))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    report = classification_report(Y_val, y_pred_valid, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)
    
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    report = classification_report(Y_test, y_pred_test, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])
    
    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_test))

def cnn_khaiii_target(data):
    X = data['clean_words_khaiii']
    Y = data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42, stratify = Y_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_tr)
    vocab_size = len(tokenizer.word_index) + 1
    data_train = tokenizer.texts_to_sequences(X_tr)
    data_valid = tokenizer.texts_to_sequences(X_val)
    data_test = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(i) for i in data_train)

    X_train = pad_sequences(data_train, maxlen = max_len)
    X_valid = pad_sequences(data_valid, maxlen = max_len)
    X_test = pad_sequences(data_test, maxlen = max_len)

    vocab_size = len(tokenizer.word_index)
    embedding_dim = 64

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(layers.Conv1D(128, 5, activation = 'relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(32, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 4)
    mc = ModelCheckpoint('Models/cnn_khaiii_binary_best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

    history = model.fit(X_train, Y_tr, epochs = 15, callbacks = [es, mc], batch_size = 16, validation_split = 0.2)

    y_pred_train = model.predict(X_train)
    score_train= [i for j in y_pred_train.tolist() for i in j]
    y_pred_train = []
    for i in score_train:
        if i > 0.5 :
            y_pred_train.append(1)
        else:
            y_pred_train.append(0)

    y_pred_valid = model.predict(X_valid)
    score_valid = [i for j in y_pred_valid.tolist() for i in j]
    y_pred_valid = []
    for i in score_valid:
        if i > 0.5 :
            y_pred_valid.append(1)
        else:
            y_pred_valid.append(0)

    y_pred_test = model.predict(X_test)
    score_test = [i for j in y_pred_test.tolist() for i in j]
    y_pred_test= []
    for i in score_test:
        if i > 0.5 :
            y_pred_test.append(1)
        else:
            y_pred_test.append(0)   

    report = classification_report(Y_tr, y_pred_train, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (10, 10))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    report = classification_report(Y_val, y_pred_valid, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)
    
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    report = classification_report(Y_test, y_pred_test, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])
    
    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_test))

def lstm_mecab_cat(data):
    data = data[data['category'] > 0].reset_index(drop = True)
    X = data['clean_words_mecab']
    Y = data['category']
    Y = data['category']-1
    X_train, X_test, Y_train, Y_te = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42, stratify = Y_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_tr)
    vocab_size = len(tokenizer.word_index) + 1
    data_train = tokenizer.texts_to_sequences(X_tr)
    data_valid = tokenizer.texts_to_sequences(X_val)
    data_test = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(i) for i in data_train)

    X_train = pad_sequences(data_train, maxlen = max_len)
    X_valid = pad_sequences(data_valid, maxlen = max_len)
    X_test = pad_sequences(data_test, maxlen = max_len)
    Y_train = to_categorical(Y_tr, num_classes = 4)
    Y_valid = to_categorical(Y_val, num_classes = 4)
    Y_test = to_categorical(Y_te, num_classes = 4)

    vocab_size = len(tokenizer.word_index)
    embedding_dim = 100
    hidden_units = 128

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Bidirectional(LSTM(hidden_units)))
    model.add(Activation('ReLU'))
    model.add(Dense(4, activation = 'softmax'))

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
    mc = ModelCheckpoint('Models/lstm_mecab_multi_best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])
    history = model.fit(X_train, Y_train, epochs = 15, callbacks = [es, mc], batch_size = 16, validation_split = 0.2)

    labels = [0, 1, 2, 3]
    y_pred_trains = model.predict(X_train)
    y_pred_trains = y_pred_trains.tolist()
    y_pred_train = []
    for score in y_pred_trains:
        y_pred_train.append(labels[np.argmax(score)])   
    y_pred_valids = model.predict(X_valid)
    y_pred_valids = y_pred_valids.tolist()
    y_pred_valid = []
    for score in y_pred_valids:
        y_pred_valid.append(labels[np.argmax(score)]) 
    y_pred_tests = model.predict(X_test)
    y_pred_tests = y_pred_tests.tolist()
    y_pred_test = []
    for score in y_pred_tests:
        y_pred_test.append(labels[np.argmax(score)])  

    report = classification_report(Y_tr, y_pred_train, labels = [0, 1, 2, 3], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (12, 12))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    report = classification_report(Y_val, y_pred_valid, labels = [0, 1, 2, 3], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    report = classification_report(Y_te, y_pred_test, labels = [0, 1, 2, 3],  target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])

    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    
    print(pd.DataFrame(report))
    print(accuracy_score(Y_te, y_pred_test))

def lstm_mecab_cat_sampled(data):
    data = data[data['category'] > 0].reset_index(drop = True)
    data_1 = data[data['category'] == 1]
    data_2 = data[data['category'] == 2].sample(n = 693, replace = False)
    data_3 = data[data['category'] == 3].sample(n = 693, replace = False)
    data_4 = data[data['category'] == 4].sample(n = 693, replace = False)
    data = pd.concat([data_1, data_2, data_3, data_4], axis = 0)
    X = data['clean_words_mecab']
    Y = data['category']
    Y = data['category']-1
    X_train, X_test, Y_train, Y_te = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42, stratify = Y_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_tr)
    vocab_size = len(tokenizer.word_index) + 1
    data_train = tokenizer.texts_to_sequences(X_tr)
    data_valid = tokenizer.texts_to_sequences(X_val)
    data_test = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(i) for i in data_train)

    X_train = pad_sequences(data_train, maxlen = max_len)
    X_valid = pad_sequences(data_valid, maxlen = max_len)
    X_test = pad_sequences(data_test, maxlen = max_len)
    Y_train = to_categorical(Y_tr, num_classes = 4)
    Y_valid = to_categorical(Y_val, num_classes = 4)
    Y_test = to_categorical(Y_te, num_classes = 4)

    vocab_size = len(tokenizer.word_index)
    embedding_dim = 100
    hidden_units = 128

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Bidirectional(LSTM(hidden_units)))
    model.add(Activation('ReLU'))
    model.add(Dense(4, activation = 'softmax'))

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
    mc = ModelCheckpoint('Models/lstm_mecab_multi_best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])
    history = model.fit(X_train, Y_train, epochs = 15, callbacks = [es, mc], batch_size = 16, validation_split = 0.2)

    labels = [0, 1, 2, 3]
    y_pred_trains = model.predict(X_train)
    y_pred_trains = y_pred_trains.tolist()
    y_pred_train = []
    for score in y_pred_trains:
        y_pred_train.append(labels[np.argmax(score)])   
    y_pred_valids = model.predict(X_valid)
    y_pred_valids = y_pred_valids.tolist()
    y_pred_valid = []
    for score in y_pred_valids:
        y_pred_valid.append(labels[np.argmax(score)]) 
    y_pred_tests = model.predict(X_test)
    y_pred_tests = y_pred_tests.tolist()
    y_pred_test = []
    for score in y_pred_tests:
        y_pred_test.append(labels[np.argmax(score)])  

    report = classification_report(Y_tr, y_pred_train, labels = [0, 1, 2, 3], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (12, 12))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    report = classification_report(Y_val, y_pred_valid, labels = [0, 1, 2, 3], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    report = classification_report(Y_te, y_pred_test, labels = [0, 1, 2, 3],  target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])

    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    
    print(pd.DataFrame(report))
    print(accuracy_score(Y_te, y_pred_test))

def lstm_mecab_onestage(data):
    data_0 = data[data['category'] == 0].sample(n = 693, replace = False)
    data_1 = data[data['category'] == 1]
    data_2 = data[data['category'] == 2].sample(n = 693, replace = False)
    data_3 = data[data['category'] == 3].sample(n = 693, replace = False)
    data_4 = data[data['category'] == 4].sample(n = 693, replace = False)
    data = pd.concat([data_0, data_1, data_2, data_3, data_4], axis = 0)
    X = data['clean_words_mecab']
    Y = data['category']
    Y = data['category']
    X_train, X_test, Y_train, Y_te = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42, stratify = Y_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_tr)
    vocab_size = len(tokenizer.word_index) + 1
    data_train = tokenizer.texts_to_sequences(X_tr)
    data_valid = tokenizer.texts_to_sequences(X_val)
    data_test = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(i) for i in data_train)

    X_train = pad_sequences(data_train, maxlen = max_len)
    X_valid = pad_sequences(data_valid, maxlen = max_len)
    X_test = pad_sequences(data_test, maxlen = max_len)
    Y_train = to_categorical(Y_tr, num_classes = 5)
    Y_valid = to_categorical(Y_val, num_classes = 5)
    Y_test = to_categorical(Y_te, num_classes = 5)

    vocab_size = len(tokenizer.word_index)
    embedding_dim = 100
    hidden_units = 128

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Bidirectional(LSTM(hidden_units)))
    model.add(Activation('ReLU'))
    model.add(Dense(5, activation = 'softmax'))

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
    mc = ModelCheckpoint('Models/lstm_mecab_multi_best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])
    history = model.fit(X_train, Y_train, epochs = 15, callbacks = [es, mc], batch_size = 16, validation_split = 0.2)

    labels = [0, 1, 2, 3, 4]
    y_pred_trains = model.predict(X_train)
    y_pred_trains = y_pred_trains.tolist()
    y_pred_train = []
    for score in y_pred_trains:
        y_pred_train.append(labels[np.argmax(score)])   
    y_pred_valids = model.predict(X_valid)
    y_pred_valids = y_pred_valids.tolist()
    y_pred_valid = []
    for score in y_pred_valids:
        y_pred_valid.append(labels[np.argmax(score)]) 
    y_pred_tests = model.predict(X_test)
    y_pred_tests = y_pred_tests.tolist()
    y_pred_test = []
    for score in y_pred_tests:
        y_pred_test.append(labels[np.argmax(score)])  

    report = classification_report(Y_tr, y_pred_train, labels = [0, 1, 2, 3, 4], target_names = ['not helpful', 'cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (12, 12))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    report = classification_report(Y_val, y_pred_valid, labels = [0, 1, 2, 3, 4], target_names = ['not helpful', 'cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    report = classification_report(Y_te, y_pred_test, labels = [0, 1, 2, 3, 4], target_names = ['not helpful', 'cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])

    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    
    print(pd.DataFrame(report))
    print(accuracy_score(Y_te, y_pred_test))

def cnn_mecab_cat(data):
    data = data[data['category'] > 0].reset_index(drop = True)
    X = data['clean_words_mecab']
    Y = data['category']
    Y = data['category']-1
    X_train, X_test, Y_train, Y_te = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42, stratify = Y_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_tr)
    vocab_size = len(tokenizer.word_index) + 1
    data_train = tokenizer.texts_to_sequences(X_tr)
    data_valid = tokenizer.texts_to_sequences(X_val)
    data_test = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(i) for i in data_train)

    X_train = pad_sequences(data_train, maxlen = max_len)
    X_valid = pad_sequences(data_valid, maxlen = max_len)
    X_test = pad_sequences(data_test, maxlen = max_len)
    Y_train = to_categorical(Y_tr, num_classes = 4)
    Y_valid = to_categorical(Y_val, num_classes = 4)
    Y_test = to_categorical(Y_te, num_classes = 4)

    vocab_size = len(tokenizer.word_index)
    embedding_dim = 100

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(layers.Conv1D(128, 5, activation = 'relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(32, activation = 'relu'))
    model.add(layers.Dense(4, activation = 'softmax'))
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
    mc = ModelCheckpoint('Models/cnn_mecab_multi_best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

    history = model.fit(X_train, Y_train, epochs = 15, callbacks = [es, mc], batch_size = 16, validation_split = 0.2)

    labels = [0, 1, 2, 3]
    y_pred_trains = model.predict(X_train)
    y_pred_trains = y_pred_trains.tolist()
    y_pred_train = []
    for score in y_pred_trains:
        y_pred_train.append(labels[np.argmax(score)])   
    y_pred_valids = model.predict(X_valid)
    y_pred_valids = y_pred_valids.tolist()
    y_pred_valid = []
    for score in y_pred_valids:
        y_pred_valid.append(labels[np.argmax(score)]) 
    y_pred_tests = model.predict(X_test)
    y_pred_tests = y_pred_tests.tolist()
    y_pred_test = []
    for score in y_pred_tests:
        y_pred_test.append(labels[np.argmax(score)])  

    report = classification_report(Y_tr, y_pred_train, labels = [0, 1, 2, 3], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (12, 12))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    report = classification_report(Y_val, y_pred_valid, labels = [0, 1, 2, 3], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    report = classification_report(Y_te, y_pred_test, labels = [0, 1, 2, 3],  target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])

    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    
    print(pd.DataFrame(report))
    print(accuracy_score(Y_te, y_pred_test))

def cnn_mecab_cat_sampled(data):
    data = data[data['category'] > 0].reset_index(drop = True)
    data_1 = data[data['category'] == 1]
    data_2 = data[data['category'] == 2].sample(n = 693, replace = False)
    data_3 = data[data['category'] == 3].sample(n = 693, replace = False)
    data_4 = data[data['category'] == 4].sample(n = 693, replace = False)
    data = pd.concat([data_1, data_2, data_3, data_4], axis = 0)

    X = data['clean_words_mecab']
    Y = data['category']
    Y = data['category']-1
    X_train, X_test, Y_train, Y_te = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42, stratify = Y_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_tr)
    vocab_size = len(tokenizer.word_index) + 1
    data_train = tokenizer.texts_to_sequences(X_tr)
    data_valid = tokenizer.texts_to_sequences(X_val)
    data_test = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(i) for i in data_train)

    X_train = pad_sequences(data_train, maxlen = max_len)
    X_valid = pad_sequences(data_valid, maxlen = max_len)
    X_test = pad_sequences(data_test, maxlen = max_len)
    Y_train = to_categorical(Y_tr, num_classes = 4)
    Y_valid = to_categorical(Y_val, num_classes = 4)
    Y_test = to_categorical(Y_te, num_classes = 4)

    vocab_size = len(tokenizer.word_index)
    embedding_dim = 100

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(layers.Conv1D(128, 5, activation = 'relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(32, activation = 'relu'))
    model.add(layers.Dense(4, activation = 'softmax'))
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
    mc = ModelCheckpoint('Models/cnn_mecab_multi_best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

    history = model.fit(X_train, Y_train, epochs = 15, callbacks = [es, mc], batch_size = 16, validation_split = 0.2)

    labels = [0, 1, 2, 3]
    y_pred_trains = model.predict(X_train)
    y_pred_trains = y_pred_trains.tolist()
    y_pred_train = []
    for score in y_pred_trains:
        y_pred_train.append(labels[np.argmax(score)])   
    y_pred_valids = model.predict(X_valid)
    y_pred_valids = y_pred_valids.tolist()
    y_pred_valid = []
    for score in y_pred_valids:
        y_pred_valid.append(labels[np.argmax(score)]) 
    y_pred_tests = model.predict(X_test)
    y_pred_tests = y_pred_tests.tolist()
    y_pred_test = []
    for score in y_pred_tests:
        y_pred_test.append(labels[np.argmax(score)])  

    report = classification_report(Y_tr, y_pred_train, labels = [0, 1, 2, 3], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (12, 12))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    report = classification_report(Y_val, y_pred_valid, labels = [0, 1, 2, 3], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    report = classification_report(Y_te, y_pred_test, labels = [0, 1, 2, 3],  target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])

    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    
    print(pd.DataFrame(report))
    print(accuracy_score(Y_te, y_pred_test))

def cnn_mecab_onestage(data):
    data_0 = data[data['category'] == 0].sample(n = 693, replace = False)
    data_1 = data[data['category'] == 1]
    data_2 = data[data['category'] == 2].sample(n = 693, replace = False)
    data_3 = data[data['category'] == 3].sample(n = 693, replace = False)
    data_4 = data[data['category'] == 4].sample(n = 693, replace = False)
    data = pd.concat([data_0, data_1, data_2, data_3, data_4], axis = 0)

    X = data['clean_words_mecab']
    Y = data['category']
    #Y = data['category']-1
    X_train, X_test, Y_train, Y_te = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42, stratify = Y_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_tr)
    vocab_size = len(tokenizer.word_index) + 1
    data_train = tokenizer.texts_to_sequences(X_tr)
    data_valid = tokenizer.texts_to_sequences(X_val)
    data_test = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(i) for i in data_train)

    X_train = pad_sequences(data_train, maxlen = max_len)
    X_valid = pad_sequences(data_valid, maxlen = max_len)
    X_test = pad_sequences(data_test, maxlen = max_len)
    Y_train = to_categorical(Y_tr, num_classes = 5)
    Y_valid = to_categorical(Y_val, num_classes = 5)
    Y_test = to_categorical(Y_te, num_classes = 5)

    vocab_size = len(tokenizer.word_index)
    embedding_dim = 100

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(layers.Conv1D(128, 5, activation = 'relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(32, activation = 'relu'))
    model.add(layers.Dense(5, activation = 'softmax'))
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
    mc = ModelCheckpoint('Models/cnn_mecab_multi_best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

    history = model.fit(X_train, Y_train, epochs = 15, callbacks = [es, mc], batch_size = 16, validation_split = 0.2)

    labels = [0, 1, 2, 3, 4]
    y_pred_trains = model.predict(X_train)
    y_pred_trains = y_pred_trains.tolist()
    y_pred_train = []
    for score in y_pred_trains:
        y_pred_train.append(labels[np.argmax(score)])   
    y_pred_valids = model.predict(X_valid)
    y_pred_valids = y_pred_valids.tolist()
    y_pred_valid = []
    for score in y_pred_valids:
        y_pred_valid.append(labels[np.argmax(score)]) 
    y_pred_tests = model.predict(X_test)
    y_pred_tests = y_pred_tests.tolist()
    y_pred_test = []
    for score in y_pred_tests:
        y_pred_test.append(labels[np.argmax(score)])  

    report = classification_report(Y_tr, y_pred_train, labels = [0, 1, 2, 3, 4], target_names = ['not helpful', 'cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (12, 12))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    report = classification_report(Y_val, y_pred_valid, labels = [0, 1, 2, 3, 4], target_names = ['not helpful', 'cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    report = classification_report(Y_te, y_pred_test, labels = [0, 1, 2, 3, 4], target_names = ['not helpful', 'cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])

    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    
    print(pd.DataFrame(report))
    print(accuracy_score(Y_te, y_pred_test))

def lstm_khaiii_cat(data):
    data = data[data['category'] > 0].reset_index(drop = True)
    X = data['clean_words_khaiii']
    Y = data['category']
    Y = data['category']-1
    X_train, X_test, Y_train, Y_te = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42, stratify = Y_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_tr)
    vocab_size = len(tokenizer.word_index) + 1
    data_train = tokenizer.texts_to_sequences(X_tr)
    data_valid = tokenizer.texts_to_sequences(X_val)
    data_test = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(i) for i in data_train)

    X_train = pad_sequences(data_train, maxlen = max_len)
    X_valid = pad_sequences(data_valid, maxlen = max_len)
    X_test = pad_sequences(data_test, maxlen = max_len)
    Y_train = to_categorical(Y_tr, num_classes = 4)
    Y_valid = to_categorical(Y_val, num_classes = 4)
    Y_test = to_categorical(Y_te, num_classes = 4)

    vocab_size = len(tokenizer.word_index)
    embedding_dim = 100
    hidden_units = 64

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Bidirectional(LSTM(hidden_units)))
    model.add(Dense(4, activation = 'softmax'))

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 4)
    mc = ModelCheckpoint('Models/lstm_khaiii_multi_best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
    history = model.fit(X_train, Y_train, epochs = 15, callbacks = [es, mc], batch_size = 32, validation_split = 0.2)

    labels = [0, 1, 2, 3]
    y_pred_trains = model.predict(X_train)
    y_pred_trains = y_pred_trains.tolist()
    y_pred_train = []
    for score in y_pred_trains:
        y_pred_train.append(labels[np.argmax(score)])   
    y_pred_valids = model.predict(X_valid)
    y_pred_valids = y_pred_valids.tolist()
    y_pred_valid = []
    for score in y_pred_valids:
        y_pred_valid.append(labels[np.argmax(score)]) 
    y_pred_tests = model.predict(X_test)
    y_pred_tests = y_pred_tests.tolist()
    y_pred_test = []
    for score in y_pred_tests:
        y_pred_test.append(labels[np.argmax(score)])  

    report = classification_report(Y_tr, y_pred_train, labels = [0, 1, 2, 3], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (12, 12))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    report = classification_report(Y_val, y_pred_valid, labels = [0, 1, 2, 3], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    report = classification_report(Y_te, y_pred_test, labels = [0, 1, 2, 3],  target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])

    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    
    print(pd.DataFrame(report))
    print(accuracy_score(Y_te, y_pred_test))

def cnn_khaiii_cat(data):
    data = data[data['category'] > 0].reset_index(drop = True)
    X = data['clean_words_khaiii']
    Y = data['category']
    Y = data['category']-1
    X_train, X_test, Y_train, Y_te = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42, stratify = Y_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_tr)
    vocab_size = len(tokenizer.word_index) + 1
    data_train = tokenizer.texts_to_sequences(X_tr)
    data_valid = tokenizer.texts_to_sequences(X_val)
    data_test = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(i) for i in data_train)

    X_train = pad_sequences(data_train, maxlen = max_len)
    X_valid = pad_sequences(data_valid, maxlen = max_len)
    X_test = pad_sequences(data_test, maxlen = max_len)
    Y_train = to_categorical(Y_tr, num_classes = 4)
    Y_valid = to_categorical(Y_val, num_classes = 4)
    Y_test = to_categorical(Y_te, num_classes = 4)

    vocab_size = len(tokenizer.word_index)
    embedding_dim = 100
    hidden_units = 64

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(layers.Conv1D(64, 5, activation = 'relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(16, activation = 'relu'))
    model.add(layers.Dense(4, activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 4)
    mc = ModelCheckpoint('Models/cnn_khaiii_multi_best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

    history = model.fit(X_train, Y_train, epochs = 15, callbacks = [es, mc], batch_size = 32, validation_split = 0.2)

    labels = [0, 1, 2, 3]
    y_pred_trains = model.predict(X_train)
    y_pred_trains = y_pred_trains.tolist()
    y_pred_train = []
    for score in y_pred_trains:
        y_pred_train.append(labels[np.argmax(score)])   
    y_pred_valids = model.predict(X_valid)
    y_pred_valids = y_pred_valids.tolist()
    y_pred_valid = []
    for score in y_pred_valids:
        y_pred_valid.append(labels[np.argmax(score)]) 
    y_pred_tests = model.predict(X_test)
    y_pred_tests = y_pred_tests.tolist()
    y_pred_test = []
    for score in y_pred_tests:
        y_pred_test.append(labels[np.argmax(score)])  

    report = classification_report(Y_tr, y_pred_train, labels = [0, 1, 2, 3], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (12, 12))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    report = classification_report(Y_val, y_pred_valid, labels = [0, 1, 2, 3], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    report = classification_report(Y_te, y_pred_test, labels = [0, 1, 2, 3],  target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])

    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    
    print(pd.DataFrame(report))
    print(accuracy_score(Y_te, y_pred_test))