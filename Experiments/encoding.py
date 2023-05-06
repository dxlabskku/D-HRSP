import pandas as pd 
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, plot_roc_curve, roc_curve, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

def og_cv_onestage_rf(data):
    data_0 = data[data['category'] == 0]
    data_1 = data[data['category'] == 1]
    data_2 = data[data['category'] == 2]
    data_3 = data[data['category'] == 3]
    data_4 = data[data['category'] == 4]
    data = pd.concat([data_0, data_1, data_2, data_3, data_4], axis = 0)
    cv = CountVectorizer(max_features = 100000)
    X = cv.fit_transform(data['clean_words_mecab'].values.astype('U')).toarray()
    labels = data['category'].tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

    clf = RandomForestClassifier() 
    clf.fit(X_tr, Y_tr)

    y_pred_tr = clf.predict(X_tr)
    report = classification_report(Y_tr, y_pred_tr, labels = [0, 1, 2, 3, 4], target_names = ['not helpful', 'cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (12, 12))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    y_pred_va = clf.predict(X_val)
    report = classification_report(Y_val, y_pred_va, labels = [0, 1, 2, 3, 4], target_names = ['not helpful', 'cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    y_pred_te = clf.predict(X_test)
    report = classification_report(Y_test, y_pred_te, labels = [0, 1, 2, 3, 4], target_names = ['not helpful', 'cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])

    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    
    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_te))

def og_cv_target_rf(data): # Simple Spacing Binary Classification

    corpus = data['cleaned'].tolist()
    cv = CountVectorizer(max_features = 100000)
    X = cv.fit_transform(corpus).toarray()
    labels = data['target'].tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

    clf = RandomForestClassifier()
    clf.fit(X_tr, Y_tr)

    y_pred_tr = clf.predict(X_tr)
    report = classification_report(Y_tr, y_pred_tr, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (10, 10))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    y_pred_va = clf.predict(X_val)
    report = classification_report(Y_val, y_pred_va, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)
    
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    y_pred_te = clf.predict(X_test)
    report = classification_report(Y_test, y_pred_te, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])
    
    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()

    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_te))

def mecab_cv_onestage_rf(data):
    data_0 = data[data['category'] == 0].sample(n = 693, replace = False)
    data_1 = data[data['category'] == 1]
    data_2 = data[data['category'] == 2].sample(n = 693, replace = False)
    data_3 = data[data['category'] == 3].sample(n = 693, replace = False)
    data_4 = data[data['category'] == 4].sample(n = 693, replace = False)
    data = pd.concat([data_0, data_1, data_2, data_3, data_4], axis = 0)

    cv = CountVectorizer(max_features = 100000)
    X = cv.fit_transform(data['clean_words_mecab'].values.astype('U')).toarray()
    labels = data['category'].tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

    clf = RandomForestClassifier()
    clf.fit(X_tr, Y_tr)

    y_pred_tr = clf.predict(X_tr)
    report = classification_report(Y_tr, y_pred_tr, labels = [0, 1, 2, 3, 4], target_names = ['not helpful', 'cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (12, 12))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    y_pred_va = clf.predict(X_val)
    report = classification_report(Y_val, y_pred_va, labels = [0, 1, 2, 3, 4], target_names = ['not helpful', 'cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    y_pred_te = clf.predict(X_test)
    report = classification_report(Y_test, y_pred_te, labels = [0, 1, 2, 3, 4], target_names = ['not helpful', 'cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])
    
    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()

    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_te))

def mecab_cv_target_rf(data): # Mecab Tokenizer Binary Classification

    cv = CountVectorizer(max_features = 100000)
    X = cv.fit_transform(data['clean_words_mecab'].values.astype('U')).toarray()
    labels = data['target'].tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

    clf = RandomForestClassifier()
    clf.fit(X_tr, Y_tr)

    y_pred_tr = clf.predict(X_tr)
    report = classification_report(Y_tr, y_pred_tr, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (10, 10))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    y_pred_va = clf.predict(X_val)
    report = classification_report(Y_val, y_pred_va, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)
    
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    y_pred_te = clf.predict(X_test)
    report = classification_report(Y_test, y_pred_te, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])
    
    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()

    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_te))

def mecab_cv_cat_rf(data): # Mecab Tokenizer Multi-class Classification

    data = data[data['category'] > 0].reset_index(drop = True)
    cv = CountVectorizer(max_features = 100000)
    X = cv.fit_transform(data['clean_words_mecab'].values.astype('U')).toarray()
    labels = data['category'].tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

    clf = RandomForestClassifier() 
    clf.fit(X_tr, Y_tr)

    y_pred_tr = clf.predict(X_tr)
    report = classification_report(Y_tr, y_pred_tr, labels = [1, 2, 3, 4], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (12, 12))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    y_pred_va = clf.predict(X_val)
    report = classification_report(Y_val, y_pred_va, labels = [1, 2, 3, 4], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    y_pred_te = clf.predict(X_test)
    report = classification_report(Y_test, y_pred_te, labels = [1, 2, 3, 4], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])

    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    
    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_te))

def khaiii_cv_onestage_rf(data):
    data_0 = data[data['category'] == 0]
    data_1 = data[data['category'] == 1]
    data_2 = data[data['category'] == 2]
    data_3 = data[data['category'] == 3]
    data_4 = data[data['category'] == 4]
    data = pd.concat([data_0, data_1, data_2, data_3, data_4], axis = 0)

    cv = CountVectorizer(max_features = 100000)
    X = cv.fit_transform(data['clean_words_khaiii'].values.astype('U')).toarray()
    labels = data['category'].tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

    clf = RandomForestClassifier() 
    clf.fit(X_tr, Y_tr)

    y_pred_tr = clf.predict(X_tr)
    report = classification_report(Y_tr, y_pred_tr, labels = [0, 1, 2, 3, 4], target_names = ['not helpful', 'cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (12, 12))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    y_pred_va = clf.predict(X_val)
    report = classification_report(Y_val, y_pred_va, labels = [0, 1, 2, 3, 4], target_names = ['not helpful', 'cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    y_pred_te = clf.predict(X_test)
    report = classification_report(Y_test, y_pred_te, labels = [0, 1, 2, 3, 4], target_names = ['not helpful', 'cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])

    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_te))
    
def khaiii_cv_target_rf(data): # Khaiii Tokenizer Binary Classification

    cv = CountVectorizer(max_features = 100000)
    X = cv.fit_transform(data['clean_words_khaiii'].values.astype('U')).toarray()
    labels = data['target'].tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

    clf = RandomForestClassifier()
    clf.fit(X_tr, Y_tr)

    y_pred_tr = clf.predict(X_tr)
    report = classification_report(Y_tr, y_pred_tr, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (10, 10))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    y_pred_va = clf.predict(X_val)
    report = classification_report(Y_val, y_pred_va, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)
    
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    y_pred_te = clf.predict(X_test)
    report = classification_report(Y_test, y_pred_te, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])
    
    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()

    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_te))

def khaiii_cv_cat_rf(data): # Khaiii Tokenizer Multi-class Classification


    data = data[data['category'] > 0].reset_index(drop = True)
    cv = CountVectorizer(max_features = 100000)
    X = cv.fit_transform(data['clean_words_khaiii'].values.astype('U')).toarray()
    labels = data['category'].tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

    clf = RandomForestClassifier() 
    clf.fit(X_tr, Y_tr)

    y_pred_tr = clf.predict(X_tr)
    report = classification_report(Y_tr, y_pred_tr, labels = [1, 2, 3, 4], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (12, 12))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    y_pred_va = clf.predict(X_val)
    report = classification_report(Y_val, y_pred_va, labels = [1, 2, 3, 4], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    y_pred_te = clf.predict(X_test)
    report = classification_report(Y_test, y_pred_te, labels = [1, 2, 3, 4], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])

    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_te))

def og_cv_target_rf_sampled(data): # Simple Spacing Binary Classification
    data_0 = data[data['target'] == 0]
    data_1 = data[data['target'] == 1]
    data_1 = data_1.sample(n = 2590, replace = False)
    data = pd.concat([data_0,data_1], axis = 0)
    corpus = data['cleaned'].tolist()
    cv = CountVectorizer(max_features = 100000)
    X = cv.fit_transform(corpus).toarray()
    labels = data['target'].tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

    clf = RandomForestClassifier()
    clf.fit(X_tr, Y_tr)

    y_pred_tr = clf.predict(X_tr)
    report = classification_report(Y_tr, y_pred_tr, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (10, 10))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    y_pred_va = clf.predict(X_val)
    report = classification_report(Y_val, y_pred_va, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)
    
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    y_pred_te = clf.predict(X_test)
    report = classification_report(Y_test, y_pred_te, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])
    
    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()

    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_te))

def mecab_cv_target_rf_sampled(data): # Mecab Tokenizer Binary Classification
    data_0 = data[data['target'] == 0]
    data_1 = data[data['target'] == 1]
    data_1 = data_1.sample(n = 2590, replace = False)
    data = pd.concat([data_0,data_1], axis = 0)
    corpus = data['cleaned'].tolist()
    cv = CountVectorizer(max_features = 100000)
    X = cv.fit_transform(data['clean_words_mecab'].values.astype('U')).toarray()
    labels = data['target'].tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

    clf = RandomForestClassifier()
    clf.fit(X_tr, Y_tr)

    y_pred_tr = clf.predict(X_tr)
    report = classification_report(Y_tr, y_pred_tr, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (10, 10))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    y_pred_va = clf.predict(X_val)
    report = classification_report(Y_val, y_pred_va, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)
    
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    y_pred_te = clf.predict(X_test)
    report = classification_report(Y_test, y_pred_te, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])
    
    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()

    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_te))

def mecab_cv_cat_rf_sampled(data): # Mecab Tokenizer Multi-class Classification
    data = data[data['category'] > 0].reset_index(drop = True)
    data_1 = data[data['category'] == 1]
    data_2 = data[data['category'] == 2].sample(n = 693, replace = False)
    data_3 = data[data['category'] == 3].sample(n = 693, replace = False)
    data_4 = data[data['category'] == 4].sample(n = 693, replace = False)
    data = pd.concat([data_1, data_2, data_3, data_4], axis = 0)
    cv = CountVectorizer(max_features = 100000)
    X = cv.fit_transform(data['clean_words_mecab'].values.astype('U')).toarray()
    labels = data['category'].tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

    clf = RandomForestClassifier() 
    clf.fit(X_tr, Y_tr)

    y_pred_tr = clf.predict(X_tr)
    report = classification_report(Y_tr, y_pred_tr, labels = [1, 2, 3, 4], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (12, 12))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    y_pred_va = clf.predict(X_val)
    report = classification_report(Y_val, y_pred_va, labels = [1, 2, 3, 4], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    y_pred_te = clf.predict(X_test)
    report = classification_report(Y_test, y_pred_te, labels = [1, 2, 3, 4], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])

    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    
    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_te))

def khaiii_cv_target_rf_sampled(data): # Khaiii Tokenizer Binary Classification
    data_0 = data[data['target'] == 0]
    data_1 = data[data['target'] == 1]
    data_1 = data_1.sample(n = 2590, replace = False)
    data = pd.concat([data_0,data_1], axis = 0)
    corpus = data['cleaned'].tolist()
    cv = CountVectorizer(max_features = 100000)
    X = cv.fit_transform(data['clean_words_khaiii'].values.astype('U')).toarray()
    labels = data['target'].tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

    clf = RandomForestClassifier()
    clf.fit(X_tr, Y_tr)

    y_pred_tr = clf.predict(X_tr)
    report = classification_report(Y_tr, y_pred_tr, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (10, 10))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    y_pred_va = clf.predict(X_val)
    report = classification_report(Y_val, y_pred_va, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)
    
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    y_pred_te = clf.predict(X_test)
    report = classification_report(Y_test, y_pred_te, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])
    
    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()

    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_te))

def khaiii_cv_cat_rf_sampled(data): # Khaiii Tokenizer Multi-class Classification

    data = data[data['category'] > 0].reset_index(drop = True)
    data_1 = data[data['category'] == 1]
    data_2 = data[data['category'] == 2].sample(n = 693, replace = False)
    data_3 = data[data['category'] == 3].sample(n = 693, replace = False)
    data_4 = data[data['category'] == 4].sample(n = 693, replace = False)
    data = pd.concat([data_1, data_2, data_3, data_4], axis = 0)
    cv = CountVectorizer(max_features = 100000)
    X = cv.fit_transform(data['clean_words_khaiii'].values.astype('U')).toarray()
    labels = data['category'].tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

    clf = RandomForestClassifier() 
    clf.fit(X_tr, Y_tr)

    y_pred_tr = clf.predict(X_tr)
    report = classification_report(Y_tr, y_pred_tr, labels = [1, 2, 3, 4], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (12, 12))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    y_pred_va = clf.predict(X_val)
    report = classification_report(Y_val, y_pred_va, labels = [1, 2, 3, 4], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    y_pred_te = clf.predict(X_test)
    report = classification_report(Y_test, y_pred_te, labels = [1, 2, 3, 4], target_names = ['cat1', 'cat2', 'cat3', 'cat4'], output_dict = True, digits = 4)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])

    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    print(pd.DataFrame(report))
    print(accuracy_score(Y_test, y_pred_te))