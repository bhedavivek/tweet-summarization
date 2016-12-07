from aylienapiclient import textapi

client = textapi.Client("4d64645a", "de2ded2e8fb83ce15fb8281eee6d7cb4")

sentiment = client.Sentiment()

print sentiment