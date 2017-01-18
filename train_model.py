from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import re
import json
from scorched import SolrInterface

path = 'twitter_data/sample.json'
file = open(path,"r")
doc_complete=[]
count=0
location = []
emoji_pattern = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]', flags=re.UNICODE)
tweets = []
for line in file:
	obj = json.loads(line)
	if 'limit' not in obj:
		tweets.append(obj)
		text = obj['text']
		text = re.sub(r'http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
		text = re.sub(r"[^a-z A-Z]",' ',text)
		text = emoji_pattern.sub(r'', text)
		doc_complete.append(text)
print 'file read'
stop = stopwords.words('english')
stop.append(u'christmas')
stop.append(u'xmas')
stop.append(u'via')
stop.append(u'yo')
stop.append(u'u')
stop.append(u'rt')
stop.append(u'as')
stop.append(u'im')
stop.append(u'amp')
stop.append(u'af')
stop.append(u'am')
stop.append(u'pm')
stop.append(u'th')
stop.append(u'gonna')
stop.append(u'follow')
stop.append(u't')
stop = set(stop)
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    temp = " ".join(i for i in punc_free.lower().split() if i not in stop)
    #normalized = " ".join(lemma.lemmatize(word) for word in temp.split())
    return temp
doc_clean = [clean(doc).split() for doc in doc_complete]

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
num_topics = 5
ldamodel = Lda(doc_term_matrix, num_topics=num_topics, id2word = dictionary, passes=10)

print 'model trained'
for topics in ldamodel.show_topics(num_topics=num_topics, num_words=5):
	print topics
lda_model_path = 'lda_model/model'
ldamodel.save(lda_model_path)
text = "Just played: A Ceremony of Carols, Op. 28: Procession - Benjamin Britten, Frances Kelly, Marie-Claire Brookshaw, Clare Wilkinson - Brit..."
text = clean(text).split()
temp = ldamodel[dictionary.doc2bow(text)]
#print temp
#terms = ldamodel.get_topic_terms(4,topn=2)
#print [dictionary[terms[i][0]] for i in range(2)]

#Posting to Solr
si = SolrInterface("http://54.202.209.219:8983/solr/IRF16P4")

for i in range(len(doc_complete)):
	probs = ldamodel[dictionary.doc2bow(clean(doc_complete[i]).split())]
	max_prob =0
	max_index = 0
	label = 0
	for j in range(len(probs)):
		if probs[j][1]>max_prob:
			max_prob = probs[j][1]
			max_index=j
			label = probs[j][0]
	terms = ldamodel.get_topic_terms(max_index,topn=2)
	topics = [dictionary[terms[k][0]] for k in range(2)]
	#print topics,"<======>",
	#print tweets[i]["text"]
	newDocument = {}
	newDocument["text"] = tweets[i]["text"]
	newDocument["id"] = tweets[i]["id"]
	if tweets[i]["place"] != None:
		newDocument["country"] = tweets[i]["place"]["country"]
		newDocument["country_code"] = tweets[i]["place"]["country_code"]
	else:
		newDocument["country"] = None
		newDocument["country_code"] = None
	newDocument["text_"+str(label)] = tweets[i]["text"]
	newDocument["topic_"+str(label)] = label
	newDocument["topics"] = topics
	newDocument["created_at"] = tweets[i]["created_at"]
	newDocument["screen_name"] = tweets[i]["user"]["screen_name"]
	newDocument["profile_image_url"] = tweets[i]["user"]["profile_image_url"]
	si.add(newDocument)