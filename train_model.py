import json
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from stop_words import get_stop_words
import string
import gensim
from gensim import corpora
import re
import tweepy
import sys

#Fetching Tweets
auth = tweepy.OAuthHandler('JwceJ31d0VEPI0SuFZjjWOGph', 'QI7ZHODhz3Qcg1crM69AIXkE6M9HxHZBRHx6yjXGOtyAk2FWRO')
auth.set_access_token('114238938-ao057C6nmCSJLIw036Cx35FGL1v4qePw3tPUNZTt', '2ELwrS8uXaCTKeVKhF4Jqa3TIdTO3tRXwZanc07OWSq9D')

api = tweepy.API(auth)

doc_complete=[]
emoji_pattern = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]', flags=re.UNICODE)
tweets = api.search(q='christmas', count=100)
for tweet in tweets:
	text = tweet.text
	text = re.sub(r'http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
	text = re.sub(r"[^a-z A-Z]",'',text)
	text = emoji_pattern.sub(r'', text)
	doc_complete.append(text)
print 'file read'
stop = stopwords.words('english')

stop.append(u'christmas')
stop.append(u'christmas')
stop.append('want')
stop.append(u'got')
stop.append(u'get')
stop.append(u'via')
stop.append(u'yo')
stop.append(u'u')
stop.append(u'just')
stop.append(u'rt')
stop.append(u'as')
stop.append(u'im')
stop.append(u'amp')
stop.append(u'af')
stop.append(u'pm')
stop.append(u'th')
stop.append(u'gonna')
stop.append(u'am')
stop.append(u'\u2026')
stop = set(stop)

exclude = set(string.punctuation) 

lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    temp = " ".join(i for i in punc_free.lower().split() if i not in stop)
    normalized = " ".join(lemma.lemmatize(word) for word in temp.split())
    #return normalized
    return temp
doc_clean = [clean(doc).split() for doc in doc_complete[0:100]]

print 'file cleaned'
# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)
dictionary.save('lda_model/dictionary')
#print dictionary

print 'dictionary created'
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=4, id2word = dictionary, passes=20)

print 
'model trained'
for topics in ldamodel.show_topics(num_topics=6, num_words=5):
	print topics
lda_model_path = 'lda_model/model'
ldamodel.save(lda_model_path)
text = "Just played: A Ceremony of Carols, Op. 28: Procession - Benjamin Britten, Frances Kelly, Marie-Claire Brookshaw, Clare Wilkinson - Brit..."
text = clean(text).split()
temp = ldamodel[dictionary.doc2bow(text)]
print temp