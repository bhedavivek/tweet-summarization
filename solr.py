import json
from scorched import SolrInterface
si = SolrInterface("http://localhost:8983/solr/IRF16P4")
file = open('twitter_data/sample.json','r')
for line in file:
	text = json.loads(line)
	if 'limit' not in text:
		newDocument = {}
		newDocument["text"] = text["text"]
		newDocument["id"] = text["id"]
		if text["place"] != None:
			newDocument["country"] = text["place"]["country"]
			newDocument["country_code"] = text["place"]["country_code"]
		else:
			newDocument["country"] = None
			newDocument["country_code"] = None
		newDocument["text_0"] = None
		newDocument["topic_label"] = 0
		newDocument["created_at"] = text["created_at"]
		newDocument["screen_name"] = text["user"]["screen_name"]
		newDocument["profile_image_url"] = text["user"]["profile_image_url"]
		si.add(newDocument)
si.optimize()