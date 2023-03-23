from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert_tokenizer import KoBERTTokenizer
from keras_preprocessing.sequence import pad_sequences


from sklearn.metrics import classification_report, plot_roc_curve, roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def mecab_kobert_target_rf(data):

    X_train, X_test, Y_train, Y_test = train_test_split(data['clean_words_mecab'], data['target'], 
                                                            test_size=0.2, random_state=42)
    train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(Y_train)], axis = 1)
    test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(Y_test)], axis = 1)

    train.columns=["document","label"]
    test.columns=["document","label"]
    sentences = train['document']
    labels = train['label'].values

    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})

    tokenized_texts = [tokenizer.encode(sent) for sent in sentences]

    MAX_LEN = 512
    input_ids = tokenized_texts
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                                                    random_state=42, test_size=0.2)	
    sentences = test['document']
    labels = test['label'].values

    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})

    tokenized_texts = [tokenizer.encode(sent) for sent in sentences]

    MAX_LEN = 512

    input_ids = tokenized_texts
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    test_inputs = input_ids
    test_labels = labels

    clf = RandomForestClassifier()
    clf.fit(train_inputs, train_labels)

    y_pred_tr = clf.predict(train_inputs)
    report = classification_report(train_labels, y_pred_tr, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits= 4)

    fig, axes = plt.subplots(nrows = 3, sharex = True)
    fig.tight_layout(pad = 2.0)
    plt.figure(figsize = (10, 10))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[0])

    y_pred_va = clf.predict(validation_inputs)
    report = classification_report(validation_labels, y_pred_va, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)
    
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[1])

    y_pred_te = clf.predict(test_inputs)
    report = classification_report(test_labels, y_pred_te, labels = [0, 1], target_names = ['not helpful', 'helpful'], output_dict = True, digits = 4)

    sns.heatmap(pd.DataFrame(report).iloc[:-1, :-2].T, annot = True, ax = axes[2])
    
    axes[0].set_title('Train Classification Report')
    axes[1].set_title('Validation Classification Report')
    axes[2].set_title('Test Classification Report')
    plt.show()
    print(pd.DataFrame(report))                                              	