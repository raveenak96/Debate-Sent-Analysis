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

#This file consists of building and testing the unsupervised model, as well as data visualization of word frequencies. A different dataset containing tweets on Bernie Sanders was used to test the model, but the methodology is the same as the main report.

def process_data(tweet) :
    #remove symbols, @s
    #remove hashtags
    #tokenize
    num_tweet = re.sub("\d+", "",tweet)
    name_tweet = ''
    for sent in nltk.sent_tokenize(num_tweet):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            try:
                if chunk.label() in ('PERSON','ORGANIZATION'):
                    if name_tweet[-1:] in ('#','@'):
                       name_tweet = name_tweet[:-1]
                    else:
                        pass
                else :
                    for c in chunk.leaves():
                        name_tweet = name_tweet + ' ' + str(c[0])
            except AttributeError:
                if (name_tweet[-1:] in punctuation and name_tweet[-1:] not in ('!','?','.','&')) or (str(chunk[0]) in punctuation and str(chunk[0])!='&') :
                    name_tweet = name_tweet + str(chunk[0])
                else :
                    name_tweet = name_tweet + ' ' + str(chunk[0])
    stopWords = stopwords.words('english')
    stopWords.extend(['$','trump','warren','sen.','senator','mayor','president','kamala','harris','silent','deleted','sanders','berniesanders','ami','klobuchar','pete','beto',"o'rourke"])
    tokenizer = CrazyTokenizer(normalize=2,hashtags=False,remove_punct=True,decontract=True,latin_chars_fix=True,ignore_stopwords=stopWords,ignore_quotes=True,remove_nonunicode=True,twitter_handles='ANOTHER_TWITTER_USER',urls='URL',pos_emojis=True,neg_emojis=True,neutral_emojis=True)
    token_tweet = tokenizer.tokenize(name_tweet)
    clean_tweet = [word.strip() for word in token_tweet if len(word)>1]
    return clean_tweet

def calc_word_freq(tweets,hashtag=False) :
    # calculate word and word co-occurrence frequencies
    co_oc = defaultdict(lambda : defaultdict(int))
    freqs = defaultdict(lambda: 0)
    for tweetlist in tweets :
        clean_list = [word for word in tweetlist if word not in ('ANOTHER_TWITTER_USER', 'URL') and word[0] != '#' and len(word)>1]
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
    # calculate word probabilites and semantic orientation of words

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

def calc_sem_orientation(tweets,pos_vocab,neg_vocab) :
    # calculate semantic orientation of a phrase and data visualization
    freqs, co_oc,terms_max = calc_word_freq(tweets['content_clean'])
    visualization(freqs[:20],terms_max[:20])
    sem_orientation = calc_probabilities(freqs,co_oc,len(tweets),pos_vocab,neg_vocab)
    tweets['semantic orientation'] = [sum(sem_orientation[word] for word in tweet) for tweet in tweets['content_clean']]
    return tweets

def visualization(freqs,terms_max) :
    #most frequent words

    plt.figure()
    plt.bar(range(len(freqs)), [item[1] for item in freqs], align='center')
    plt.xticks(range(len(freqs)), [item[0] for item in freqs], rotation=75)
    plt.title("Bernie Sanders Most Frequent Words")
    plt.tight_layout()
    plt.show(block=False)

    plt.figure()
    plt.bar(range(len(terms_max)), [item[1] for item in terms_max], align='center')
    plt.xticks(range(len(terms_max)), [item[0] for item in terms_max], rotation=75)
    plt.title("Bernie Sanders Most Frequent Co-ocurring Words")
    plt.tight_layout()
    plt.show(block=False)

def main(argv=None) :

    pd.options.mode.chained_assignment = None
    tweets_path = "./data/sanders3.csv"
    file = io.open(tweets_path,encoding="utf8")
    tweets = pd.read_csv(file, sep=',',encoding='utf8',
                           usecols=['handle','content','retweets','favorites','date'],
                           dtype={'handle':str,'content':str,'retweets':str,'favorites':str,'date':str},na_values={'retweets':'','favorites':'','content':['[]','']})
    tweets.dropna(subset=['content'],axis=0)
    tweets['content_clean'] = [process_data(tweet) for tweet in tweets['content']]
    tweets = tweets[tweets['content_clean'].str.len() != 0]

    """Using opinion lexicon from below paper
    Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." 
       Proceedings of the ACM SIGKDD International Conference on Knowledge 
       Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, 
       Washington, USA"""
    pos_file = open('positive-words.txt')
    neg_file = open('negative-words.txt')
    pos_vocab = [line.rstrip('\r\n') for line in pos_file]
    neg_vocab = [line.rstrip('\r\n') for line in neg_file]

    tweets = calc_sem_orientation(tweets,pos_vocab,neg_vocab)
    #Simply determining which tweets are positive, negative, or neutral
    pos_tweets = tweets[tweets['semantic orientation']>10.0]
    neutral_tweets = tweets[tweets['semantic orientation'].between(-10.0,10.0)]
    neg_tweets = tweets[tweets['semantic orientation']<-10.0]


if __name__ == '__main__':
    main()
