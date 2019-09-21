import pandas as pd
import numpy as np
import nltk
from nltk.tokenize.casual import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
from redditscore.tokenizer import CrazyTokenizer
import io
from string import punctuation
import matplotlib
import matplotlib.pyplot as plt
from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig
from keras import models,layers
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding,LSTM,Dense,Dropout,GRU,Flatten,Convolution1D
from keras.preprocessing import sequence
from keras import Sequential,regularizers
from keras.callbacks import ModelCheckpoint
import pickle
from xlrd import xldate_as_tuple
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from collections import Counter
from collections import defaultdict
import math

### This file consists of intro data analysis and visualization, applying the models to make predictions about sentiment, and visualization of the predictions. See report for details and results

def replaceURL(tweet):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
    urls = urls + re.findall('pic.twitter(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
    for url in urls :
        tweet = tweet.replace(url,' ')
    return tweet
def process_data(tweet)  :

    letterText = re.sub("\d+", "",tweet)
    stopWords = list(STOPWORDS)
    stopWords.extend(['demdebate','re','campaign','senator','sen','mayor','president','trump','RT','bernie','warren','kamala','buttigieg','castro','beto','klobuchar','joe','rogan','elizabeth','sander','sanders','candidate','utm','source','harris','biden','debate','people'])

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
    url_tweet = replaceURL(name_tweet)
    per_str = re.sub(r"[^a-zA-Z0-9 @]",' ',url_tweet)
    tokenizer = CrazyTokenizer(normalize=2, hashtags='', remove_punct=True, decontract=True, latin_chars_fix=True,
                               ignore_stopwords=stopWords, ignore_quotes=True, remove_nonunicode=True, twitter_handles='',urls='URL',pos_emojis=True,neg_emojis=True,neutral_emojis=True)
    token_tweet = tokenizer.tokenize(per_str)

    clean_tweet = [word.strip() for word in token_tweet if len(word) > 1]
    return " ".join(clean_tweet)

def data_prep(dataset,tokenizer,max_seq_length) :

    for i,row in dataset.iterrows() :
        val = row['TextOrReply']
        if(row['TextOrReply'].find('Replying to',0,11)!=-1) :
            temp_reply = row['TextOrReply']
            row['TextOrReply'] = row['Reply']
            row['Reply'] = temp_reply

    clean_data = dataset.rename(columns={'TextOrReply':'Tweet'}).dropna(subset=['Tweet'])
    clean_data['Tweet'] = [process_data(tweet) for tweet in clean_data['Tweet']]
    stemmer = SnowballStemmer('english')
    clean_data['Tweet_Stemmed'] = [" ".join([stemmer.stem(word) for word in tweet.split()]) for tweet in clean_data['Tweet']]
    clean_data.dropna(subset=['Tweet'],inplace=True)

    clean_data['Date'] = clean_data['Date'].apply(lambda x: (x + ' 2019'))
    clean_data['Date'] = pd.to_datetime(clean_data['Date'])

    sequences = tokenizer.texts_to_sequences(clean_data['Tweet_Stemmed'])
    sequences = sequence.pad_sequences(sequences, maxlen=max_seq_length, padding='post')


    return clean_data,sequences

def calc_word_freq(tweets,hashtag=False) :
    #calculate word and word co-occurrence frequencies

    co_oc = defaultdict(lambda : defaultdict(int))
    freqs = defaultdict(lambda: 0)
    for tweetlist in tweets :
        clean_list = [word for word in tweetlist.split() if word not in ('ANOTHER_TWITTER_USER', 'URL') and word[0] != '#' and len(word)>1]
        for word in clean_list :
            freqs[word] += 1
        for i in range(len(clean_list)-1) :
            for j in range(i+1,len(clean_list)) :
                w1,w2 = sorted([clean_list[i],clean_list[j]])
                if w1!=w2:
                    co_oc[w1][w2] += 1
    co_oc_max = []
    for t1 in co_oc :
        t1_max_terms = sorted(co_oc[t1].items(), key=lambda t: t[1], reverse=True)[:5]
        for t2, t2_count in t1_max_terms :
            co_oc_max.append(((t1,t2),t2_count))
    terms_max = sorted(co_oc_max,key=lambda co_ocs: co_ocs[1],reverse=True)
    sortedfreq = sorted(freqs.items(),key=lambda word: word[1],reverse=True)
    return sortedfreq,co_oc,terms_max

def calc_probabilities(freqs,co_oc,num_docs,pos_vocab,neg_vocab) :
    #calculate word probabilites and semantic orientation of words

    p_t = {}
    p_t_co_oc = defaultdict(lambda: defaultdict(int))
    for term,freq in freqs :
        p_t[term] = freq/num_docs
        for t2 in co_oc[term] :
            p_t_co_oc[term][t2] = co_oc[term][t2] / num_docs
    pmi = defaultdict(lambda : defaultdict(int))
    for t1,prob in p_t.items() :
        for t2 in co_oc[t1] :
            pmi[t1][t2] = math.log2(p_t_co_oc[t1][t2] / (p_t[t1]  * p_t[t2]))
    sem_orientation = defaultdict(lambda: 0)
    for t1,prob in p_t.items() :
        pos = sum(pmi[t1][tp] for tp in pos_vocab)
        neg = sum(pmi[t1][tn] for tn in neg_vocab)
        sem_orientation[t1] = pos-neg
    return sem_orientation

def calc_sem_orientation(tweets,pos_vocab,neg_vocab,name,color) :
    #calculate semantic orientation of a phrase and data visualization

    freqs, co_oc,terms_max = calc_word_freq(tweets['Tweet'])
    visualization(freqs[:20],terms_max[:20],name,color)
    probs = calc_probabilities(freqs,co_oc,len(tweets),pos_vocab,neg_vocab)
    sem_orientation = [sum(probs[word] for word in tweet.split()) for tweet in tweets['Tweet']]
    return sem_orientation

def visualization(freqs,terms_max,name,color) :
    #Most freq words bar graph
    plt.figure()
    plt.bar(range(len(freqs)), [item[1] for item in freqs], align='center',color=color)
    plt.xticks(range(len(freqs)), [item[0] for item in freqs], rotation=75)
    plt.title(name + " Most Frequent Words")
    plt.tight_layout()
    plt.show(block=False)

    #Most frequent co-occurring words bar graph
    plt.figure()
    plt.bar(range(len(terms_max)), [item[1] for item in terms_max], align='center',color=color)
    plt.xticks(range(len(terms_max)), [item[0] for item in terms_max], rotation=75)
    plt.title(name + " Most Frequent Co-occurring Words")
    plt.tight_layout()
    plt.show(block=False)
    return

def visualization2(cand_dataset,name,color) :
    #wordcloud
    dataset = cand_dataset
    wordcloud = WordCloud(background_color='white').generate(' '.join(dataset['Tweet']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(name+'Wordcloud')
    plt.axis("off")
    plt.show()

    #Pie charts of sentiment percentages
    counts = dataset['US_Pred'].value_counts(normalize=True)
    percents = [x * 100 for x in list(counts)]
    labels = ['Negative', 'Positive']
    fig1, ax1 = plt.subplots()
    ax1.pie(percents, labels=labels, autopct='%1.1f%%', startangle=90,explode=[.02,.02],colors=['orangered','deepskyblue'])
    ax1.axis('equal')
    plt.title(name + ' Twitter Sentiment 7/22-8/6 (Unsupervised)')
    plt.show()

    counts = dataset['S_Pred'].value_counts(normalize=True)
    percents = [x * 100 for x in list(counts)]
    fig2, ax1 = plt.subplots()
    ax1.pie(percents, labels=labels, autopct='%1.1f%%', startangle=90,explode=[.02,.02],colors=['orangered','deepskyblue'])
    ax1.axis('equal')
    plt.title(name + ' Twitter Sentiment 7/22-8/6 (Supervised)')
    plt.show()

    #Plotting sentiment over days
    day_group = dataset.groupby(['Date'])
    dates = []
    pos_percents_us = []
    for day, group in day_group:
        dates.append(day.strftime("%b %d"))
        pos_perc = list(group['US_Pred'].value_counts(normalize=True))[1] * 100
        pos_percents_us.append(pos_perc)
    figure = plt.figure()
    plt.title(name + ' Twitter Sentiment Over Time (Unsupervised)')
    plt.xlabel('Date')
    plt.ylabel('Positive tweet percentage')
    plt.plot(dates, pos_percents_us,color=color)

    pos_percents_s = []
    for day, group in day_group:
        pos_perc = list(group['S_Pred'].value_counts(normalize=True))[1] * 100
        pos_percents_s.append(pos_perc)
    figure = plt.figure()
    plt.title(name+' Twitter Sentiment Over Time (Supervised)')
    plt.xlabel('Date')
    plt.ylabel('Positive tweet percentage')
    plt.ylim(5,35)
    plt.plot(dates, pos_percents_s,color=color)

    #most frequent users tweeting bar graph
    user_group = dataset['User'].value_counts()[:15]
    figure = plt.figure()
    plt.bar(user_group.index, list(user_group),color=color,align='center')
    plt.title(name+' Most Frequent Users Tweeting')
    plt.xlabel('Twitter User')
    plt.ylabel('Number of tweets')
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()

    return dates,pos_percents_us,pos_percents_s

def unsupervised(dataset,name,color) :
    sem_weights = calc_sem_orientation(dataset, pos_vocab, neg_vocab,name,color)
    sem_orient = [int(orient > -50) for orient in sem_weights]
    return sem_orient




################################### Start Main ##################################################################

#Read data, concat files
s_raw_1 = pd.read_csv('data/Sanders_0731_0806_Dates.csv')
s_raw_2 = pd.read_csv('data/Sanders_0722_0730.csv')
s_raw = pd.concat([s_raw_1,s_raw_2],ignore_index=True)

w_raw = pd.read_csv('data/Warren_0722_0806_Dates.csv')

h_raw_1 = pd.read_csv('data/Harris_0731_0806_Dates.csv')
h_raw_2 = pd.read_csv('data/Harris_0722_0730.csv')
h_raw = pd.concat([h_raw_1,h_raw_2],ignore_index=True)

b_raw_1 = pd.read_csv('data/Biden_0727_0806_Dates.csv')
b_raw_2 = pd.read_csv('data/Biden_0722_0726.csv')
b_raw = pd.concat([b_raw_1,b_raw_2],ignore_index=True)


#tokenize sequences
with open('tokenizer.pickle', 'rb') as handle:
    t = pickle.load(handle)
word_index = t.word_index
max_seq_length = 28

#Prepare data
#"""
s_clean,s_sequences = data_prep(s_raw,t,max_seq_length)
w_clean,w_sequences = data_prep(w_raw,t,max_seq_length)
h_clean,h_sequences = data_prep(h_raw,t,max_seq_length)
b_clean,b_sequences = data_prep(b_raw,t,max_seq_length)

#Dumping to pickle for faster testing
pickle.dump(s_clean,open('data/s_clean.pkl','wb'))
pickle.dump(w_clean,open('data/w_clean.pkl','wb'))
pickle.dump(h_clean,open('data/h_clean.pkl','wb'))
pickle.dump(b_clean,open('data/b_clean.pkl','wb'))
pickle.dump(s_sequences,open('data/s_sequences.pkl','wb'))
pickle.dump(w_sequences,open('data/w_sequences.pkl','wb'))
pickle.dump(h_sequences,open('data/h_sequences.pkl','wb'))
pickle.dump(b_sequences,open('data/b_sequences.pkl','wb'))

#"""

#"""
s_clean = pd.read_pickle('data/s_clean.pkl').reset_index()
w_clean = pd.read_pickle('data/w_clean.pkl')
h_clean = pd.read_pickle('data/h_clean.pkl').reset_index()
b_clean = pd.read_pickle('data/b_clean.pkl').reset_index()
s_sequences = pd.read_pickle('data/s_sequences.pkl')
w_sequences = pd.read_pickle('data/w_sequences.pkl')
h_sequences = pd.read_pickle('data/h_sequences.pkl')
b_sequences = pd.read_pickle('data/b_sequences.pkl')

#"""

############################################### Supervised Predictions #######################################
#Loading trained model and making predictions
embedding_dim=150
rnn = Sequential()
embedding = Embedding(len(word_index) + 1, embedding_dim, input_length=max_seq_length)
rnn.add(embedding)
rnn.add(LSTM(50,recurrent_regularizer=regularizers.l2(.01),dropout=0.7))
rnn.add(Dense(1, activation='relu'))
rnn.load_weights('models/RNNModel.hdf5')
rnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

s_predictions = pd.concat([s_clean,pd.Series(rnn.predict_classes(s_sequences)[:,0])],axis=1).rename(columns={0:'S_Pred'})
w_predictions = pd.concat([w_clean,pd.Series(rnn.predict_classes(w_sequences)[:,0])],axis=1).rename(columns={0:'S_Pred'})
h_predictions = pd.concat([h_clean,pd.Series(rnn.predict_classes(h_sequences)[:,0])],axis=1).rename(columns={0:'S_Pred'})
b_predictions = pd.concat([b_clean,pd.Series(rnn.predict_classes(b_sequences)[:,0])],axis=1).rename(columns={0:'S_Pred'})


############################################# Unsupervised Predictions #########################################
pos_file = open('positive-words.txt')
neg_file = open('negative-words.txt')
pos_vocab = [line.rstrip('\r\n') for line in pos_file]
neg_vocab = [line.rstrip('\r\n') for line in neg_file]

s_predictions = pd.concat([s_predictions,pd.Series(unsupervised(s_clean,'Bernie Sanders','deepskyblue'))],axis=1).rename(columns={0:'US_Pred'})
w_predictions = pd.concat([w_predictions,pd.Series(unsupervised(w_clean,'Elizabeth Warren','limegreen'))],axis=1).rename(columns={0:'US_Pred'})
h_predictions = pd.concat([h_predictions,pd.Series(unsupervised(h_clean,'Kamala Harris','mediumorchid'))],axis=1).rename(columns={0:'US_Pred'})
b_predictions = pd.concat([b_predictions,pd.Series(unsupervised(b_clean,'Joe Biden','orange'))],axis=1).rename(columns={0:'US_Pred'})



############################################ More Data Visualization ####################################################
dates,s_percents_us,s_percents_s = visualization2(s_predictions,'Bernie Sanders','deepskyblue')
dates,w_percents_us,w_percents_s = visualization2(w_predictions,'Elizabeth Warren','limegreen')
dates,h_percents_us,h_percents_s = visualization2(h_predictions,'Kamala Harris','mediumorchid')
dates,b_percents_us,b_percents_s = visualization2(b_predictions,'Joe Biden','orange')

#Comparing sentiment over time line graph
figure = plt.figure()
plt.title(' Twitter Sentiment Over Time (Supervised)')
plt.xlabel('Date')
plt.ylabel('Positive tweet percentage')
plt.plot(dates, s_percents_s,color='deepskyblue',label='Bernie Sanders',linewidth=2)
plt.plot(dates, w_percents_s,color='limegreen',label='Elizabeth Warren',linewidth=2)
plt.plot(dates, h_percents_s,color='mediumorchid',label='Kamal Harris',linewidth=2)
plt.plot(dates, b_percents_s,color='orange',label='Joe Biden',linewidth=2)
plt.legend()
plt.show()

figure = plt.figure()
plt.title(' Twitter Sentiment Over Time (Unsupervised)')
plt.xlabel('Date')
plt.ylabel('Positive tweet percentage')
plt.plot(dates, s_percents_us,color='deepskyblue',label='Bernie Sanders',linewidth=2)
plt.plot(dates, w_percents_us,color='limegreen',label='Elizabeth Warren',linewidth=2)
plt.plot(dates, h_percents_us,color='mediumorchid',label='Kamal Harris',linewidth=2)
plt.plot(dates, b_percents_us,color='orange',label='Joe Biden',linewidth=2)
plt.legend()
plt.show()

#Number of tweets scraped per candidate
lengths = [len(s_predictions),len(w_predictions),len(h_predictions),len(b_predictions)]
cands = ['Sanders','Warren','Harris','Biden']

fig,ax = plt.subplots()
ax.bar(cands,lengths,align='center')
for i,v in enumerate(lengths) :
    ax.text(i-.18,v,str(v),color='blue')
plt.title('Number of Tweets Scraped')
plt.xlabel('Candidate')
plt.ylabel('Num Tweets')
plt.bar(cands,lengths,align='center')

#Overall Pos Tweet Percentages bar graph
s_counts = s_predictions['US_Pred'].value_counts(normalize=True)
w_counts = w_predictions['US_Pred'].value_counts(normalize=True)
h_counts = h_predictions['US_Pred'].value_counts(normalize=True)
b_counts = b_predictions['US_Pred'].value_counts(normalize=True)

s_counts_S = s_predictions['S_Pred'].value_counts(normalize=True)
w_counts_S = w_predictions['S_Pred'].value_counts(normalize=True)
h_counts_S = h_predictions['S_Pred'].value_counts(normalize=True)
b_counts_S = b_predictions['S_Pred'].value_counts(normalize=True)

pos_percent = [s_counts[1]*100,w_counts[1]*100,h_counts[1]*100,b_counts[1]*100]
plt.figure()
plt.bar(range(4), pos_percent, align='center',color='deepskyblue')
plt.xticks(range(4), ['Bernie Sanders','Elizabeth Warren','Kamala Harris','Joe Biden'], rotation=75)
plt.xlabel('Candidate')
plt.ylabel('Positive Tweet Percentage')
plt.title("Candidate Positive Tweet Percentages 7/22-8/06 (Unsupervised)")
plt.tight_layout()
plt.show(block=False)

pos_percent = [s_counts_S[1]*100,w_counts_S[1]*100,h_counts_S[1]*100,b_counts_S[1]*100]
plt.figure()
plt.bar(range(4), pos_percent, align='center',color='deepskyblue')
plt.xticks(range(4), ['Bernie Sanders','Elizabeth Warren','Kamala Harris','Joe Biden'], rotation=75)
plt.xlabel('Candidate')
plt.ylabel('Positive Tweet Percentage')
plt.title("Candidate Positive Tweet Percentages 7/22-8/06 (Supervised)")
plt.tight_layout()
plt.show(block=False)



#Other graphs, etc. not used
####################################### Most common words ###########################################################
#most common words, how they change over the days
day_group_clean = w_clean.groupby(['Date'])
dates = []
com_words = []
for name,group in day_group_clean:
    dates.append(name)
    all_words = ' '.join(group['Tweet']).split()
    freq = nltk.FreqDist(all_words)
    com_words.append(freq.most_common(20))



