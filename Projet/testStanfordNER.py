import nltk
import os
os.environ['JAVAHOME'] =  "C:/Program Files/Java/jdk-17.0.4.1/bin/java.exe"
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.chunk import conlltags2tree
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

st = StanfordNERTagger('./stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz',
					   './stanford-ner-2020-11-17/stanford-ner.jar',
					   encoding='utf-8')

text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'

tokenized_text = word_tokenize(text)
classified_text = st.tag(tokenized_text)

print(classified_text)