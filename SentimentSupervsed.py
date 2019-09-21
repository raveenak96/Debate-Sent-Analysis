import pandas as pd
import nltk
from nltk.tokenize.casual import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
from redditscore.tokenizer import CrazyTokenizer
import io
from string import punctuation
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,f1_score
import numpy as np
import pickle
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from keras import backend as K
import calendar
import time
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,GradientBoostingClassifier
from keras import models,layers
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding,LSTM,Dense,Dropout,GRU,Flatten,Convolution1D
from keras.preprocessing import sequence
from keras import Sequential,regularizers
from keras.callbacks import ModelCheckpoint

#This file consists of data cleanup of the Sentiment140 dataset, testing with basic ML Models, and training the RNN. See report for details and results

def process_data(tweet)  :

    letterText = re.sub("\d+", "",tweet)
    stopWords = list(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    stopWords.extend(['senator','sen.','mayor','president','trump','RT','bernie'])

    name_tweet = ""
    for sent in nltk.sent_tokenize(letterText):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            try:
                if chunk.label() in ('PERSON', 'ORGANIZATION'):
                    if name_tweet[-1:] in ('@','#') or str(chunk[0]) == 'GOPDebates':
                        name_tweet = name_tweet[:-1]
                    else:
                        pass
                else:
                    for c in chunk.leaves():
                        name_tweet = name_tweet + ' ' + str(c[0])
            except AttributeError:
                if (name_tweet[-1:] in punctuation and name_tweet[-1:] not in ('!', '?', '.', '&')) or (
                        str(chunk[0]) in punctuation and str(chunk[0]) not in  ('&','#','@')):
                    name_tweet = name_tweet + str(chunk[0])
                else:
                    name_tweet = name_tweet + ' ' + str(chunk[0])

    tokenizer = CrazyTokenizer(normalize=2, hashtags='', remove_punct=True, decontract=True, latin_chars_fix=True,
                               ignore_stopwords=stopWords, ignore_quotes=True, remove_nonunicode=True, twitter_handles='',urls='URL',pos_emojis=True,neg_emojis=True,neutral_emojis=True)
    token_tweet = tokenizer.tokenize(name_tweet)
    clean_tweet = [stemmer.stem(word.strip()) for word in token_tweet if len(word) > 1]
    return " ".join(clean_tweet)

def todense(x) :
    return x.todense()

def read_glove_file(train_text,test_text,glove_path,encoding):
    #Preparing GloVe file
    glove_data = {}
    all_words_corpus = set(word for claim_words in train_text for word in claim_words.split() )
    all_words_corpus.add(word for claim_words in test_text for word in claim_words.split())


    with open(glove_path, 'rb') as inputFile:
        for line in inputFile:
            parts = line.split()
            word = parts[0].decode(encoding)

            if (word in all_words_corpus):
                nums = np.array(parts[1:], dtype=np.float32)
                row_range = np.amax(nums) - np.amin(nums)
                row_min = np.amin(nums)
                row_avg = np.average(nums)
                row_std = np.std(nums)
                glove_data[word] = (nums + abs(row_range) + row_avg ) / row_std

    return glove_data,all_words_corpus

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def main(argv=None) :
    #"""
    train_path = 'data/train400.csv'
    test_path = 'data/test400.csv'
    train = pd.read_csv(train_path,usecols=['text','sentiment'],dtype=str,encoding='utf-8')
    test = pd.read_csv(test_path, usecols=['text', 'sentiment'],dtype=str,encoding='utf-8')


    train_processed = [process_data(tweet) for tweet in train['text']]
    test_processed = [process_data(tweet) for tweet in test['text']]

    train['text'] = train_processed
    test['text'] = test_processed

    y_train = train['sentiment']
    y_test = test['sentiment']
    X_train = train.drop(columns=['sentiment'])
    X_test = test.drop(columns=['sentiment'])

    le = LabelEncoder()
    y_train = le.fit_transform(y_train).reshape(-1,1)
    y_test = le.transform(y_test).reshape(-1,1)

    #Dumping to pickle for faster training
    X_train.to_pickle('data/x_train400.pkl')
    X_test.to_pickle('data/x_test400.pkl')
    pickle.dump(y_train,open('data/y_train400.pkl','wb'))
    pickle.dump(y_test,open('data/y_test400.pkl','wb'))
    #"""

    """
    X_train = pd.read_pickle('data/x_train400.pkl')
    X_test = pd.read_pickle('data/x_test400.pkl')
    y_train = pickle.load(open('data/y_train400.pkl','rb'))
    y_test = pickle.load(open('data/y_test400.pkl','rb'))

    train = pd.DataFrame(zip(X_train['text'],y_train[:,0]))
    test = pd.DataFrame(zip(X_test['text'], y_test[:, 0]))
    """

    train.replace('',np.nan,inplace=True)
    test.replace('', np.nan, inplace=True)
    train.dropna(axis=0,subset=[0],inplace=True)
    test.dropna(axis=0,subset=[0],inplace=True)

    X_train = train.loc[:,0]
    y_train = train.loc[:,1]
    X_test = test.loc[:,0]
    y_test = test.loc[:,1]


    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#################################### Trying basic ML Models ####################################################################################################

    """lsvc = Pipeline([('countvec',TfidfVectorizer(ngram_range=(1,2))),('svc',LinearSVC())])

    params={
            "svc__C": [1e-1,0.05,0.2,0.7, 0.9,1e-3, 1e-5]}

    gs_lsvc = GridSearchCV(lsvc,params,scoring=['accuracy','f1_micro'],refit='accuracy',cv=5,verbose=2,n_jobs=-1)
    gs_lsvc.fit(X_train['text'],y_train)

    lsvc_model = gs_lsvc.best_estimator_
    y_pred = lsvc_model.predict(X_test['text'])
    f1 = f1_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)

    svc = Pipeline([('tfidf', TfidfVectorizer()), ('svc', SVC())])

    params = {
        "svc__C": [1e-1,0.05,0.2,0.7, 1e-3, 1e-5],
        "svc__degree": [1,2,3]

    }

    gs_svc = GridSearchCV(svc, params, scoring=['accuracy', 'f1_micro'], refit='accuracy', cv=10, verbose=2, n_jobs=-1)
    gs_svc.fit(X_train['text'], y_train.reshape(-1, 1))

    svc_model = gs_svc.best_estimator_
    y_pred = svc_model.predict(X_test['text'])
    f1 = f1_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)

    gnb= Pipeline([('tfidf',TfidfVectorizer()),FunctionTransformer(todense, accept_sparse=True), ('mnb',GaussianNB())])
    params={"mnb__var_smoothing": [1e-1,0.05,0.2,0.7, 1e-3, 1e-5]}

    gs_gnb = GridSearchCV(gnb, params, scoring=['accuracy', 'f1_macro'], refit='accuracy', cv=10, verbose=4, n_jobs=-1)
    gs_gnb.fit(X_train['text'],y_train)

    gnb_model = gs_gnb.best_estimator_
    y_pred = gnb_model.predict(X_test['text'])
    f1 = f1_score(y_test,y_pred,average='micro')
    accuracy = accuracy_score(y_test,y_pred)

    bnb = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2))), ('bnb', BernoulliNB())])

    params = {
        "bnb__alpha": [10,5,3,1,.1,.01,.05,.005,.001]

    }

    gs_bnb = GridSearchCV(bnb, params, scoring=['accuracy', 'f1_micro'], refit='accuracy', cv=10, verbose=4, n_jobs=-1)
    gs_bnb.fit(X_train['text'], y_train)

    bnb_model = gs_bnb.best_estimator_
    y_pred = bnb_model.predict(X_test['text'])
    f1 = f1_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)

    rf = Pipeline([('tfidf', TfidfVectorizer()), ('ranf', RandomForestClassifier())])

    params = {
        "ranf__n_estimators": [10, 20,50,60],
        "ranf__max_depth": [2,6,8,10]

    }

    gs_rf = GridSearchCV(rf, params, scoring=['accuracy', 'f1_micro'], refit='accuracy', cv=10, verbose=4, n_jobs=-1)
    gs_rf.fit(X_train['text'], y_train)

    rf_model = gs_rf.best_estimator_
    y_pred = rf_model.predict(X_test['text'])
    f1 = f1_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)

    lr = Pipeline([('tfidf', TfidfVectorizer()), ('log', LogisticRegression(multi_class='multinomial'))])

    params = {
        "log__C": [0.7],
        "log__solver": ['newton-cg'],
        "log__penalty": ['l2']

    }

    gs_lr = GridSearchCV(lr, params, scoring=['accuracy', 'f1_micro'], refit='accuracy', cv=10, verbose=4, n_jobs=-1)
    gs_lr.fit(X_train['text'], y_train)

    lr_model = gs_lr.best_estimator_
    y_pred = lr_model.predict(X_test['text'])
    f1 = f1_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)

    gb = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2))), ('gb', GradientBoostingClassifier())])

    params = {
        "gb__learning_rate":[10,5,1,.1,.5,.001,.005,.0005],
        "gb__n_estimators": [100,60,50,10,5],
        "gb__max_depth" : [3,4,5]

    }

    gs_gb = GridSearchCV(gb, params, scoring=['accuracy', 'f1_micro'], refit='accuracy', cv=10, verbose=4, n_jobs=-1)
    gs_gb.fit(X_train['text'], y_train)

    gb_model = gs_gb.best_estimator_
    y_pred = gb_model.predict(X_test['text'])
    f1 = f1_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)

    dt= Pipeline([('tfidf', TfidfVectorizer()), ('dec', DecisionTreeClassifier())])

    params = {
        "dec__splitter":['best','random'],
        "dec__max_depth": [2,6,8,10]

    }

    gs_dt = GridSearchCV(dt, params, scoring=['accuracy', 'f1_micro'], refit='accuracy', cv=10, verbose=4, n_jobs=-1)
    gs_dt.fit(X_train['text'], y_train)

    dt_model = gs_dt.best_estimator_
    y_pred = dt_model.predict(X_test['text'])
    f1 = f1_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)

    vc = Pipeline([('tfidf',TfidfVectorizer()),('vc',VotingClassifier([('lr',LogisticRegression(C=0.7,penalty='l2',solver='newton-cg')),('lsvc',SVC(kernel='linear',probability=True,C=.2))],voting='soft'))])
    vc_model = vc.fit(X_train['text'],y_train)
    y_pred = vc_model.predict(X_test['text'])
    f1 = f1_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)


    train = list(zip(X_train['text'],y_train[:,0]))
    test = list(zip(X_test['text'], y_test[:, 0]))
    cl = NaiveBayesClassifier(train)
    cl.accuracy(test)"""

############################### RNN ####################################
    #Define parameters
    dict_size = 20000
    embedding_dim=150

    #read glove file
    #glove_path = 'Glove_Vectors/glove.twitter.27B.' + str(embedding_dim) + 'd.txt'
    #glove,corpus = read_glove_file(X_train,X_test,glove_path,'utf-8')

    #tokenize sequences
    t = Tokenizer(split=" ",oov_token='OOV')
    t.fit_on_texts(X_train)
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
    word_index = t.word_index
    X_train_tok = t.texts_to_sequences(X_train)
    X_test_tok  = t.texts_to_sequences(X_test)


    #pad sequences
    max_seq_length = max(set(len(seq) for seq in X_train_tok))
    X_train = sequence.pad_sequences(X_train_tok,maxlen=max_seq_length,padding='post')
    X_test = sequence.pad_sequences(X_test_tok, maxlen=max_seq_length,padding='post')

    #Embedding matrix if we want to use GloVe
    """embedding_matrix = np.zeros((len(word_index)+1,embedding_dim))
    for word,i in word_index.items() :
        try :
            embedding_matrix[i,:] = glove[word]
        except KeyError :
            continue"""

    batch_size = 1000
    num_epochs = 20
    val_size = 20000

    #build model
    rnn = Sequential()
    #embedding = Embedding(len(word_index)+1,embedding_dim,weights=[embedding_matrix],input_length=max_seq_length,trainable=False)
    embedding = Embedding(len(word_index) + 1, embedding_dim, input_length=max_seq_length)
    rnn.add(embedding)
    rnn.add(LSTM(50,recurrent_regularizer=regularizers.l2(.01),dropout=0.7))
    rnn.add(Dense(1, activation='relu'))
    print(rnn.summary())

    #train
    rnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy',f1_m,precision_m,recall_m])
    weight_path = 'models/best_model_' + str(calendar.timegm(time.gmtime())) + '.hdf5'
    checkpoint = ModelCheckpoint(weight_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    X_valid,y_valid = X_train[:val_size],y_train[:val_size]
    X_train2,y_train2 = X_train[val_size:],y_train[val_size:]

    rnn.fit(X_train2,y_train2,validation_data=(X_valid,y_valid),batch_size=batch_size,epochs=num_epochs,verbose=2,callbacks=callbacks_list)

    print(rnn.evaluate(X_test,y_test,batch_size=1000))

    return

if __name__ =='__main__' :
    main()