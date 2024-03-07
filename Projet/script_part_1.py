import nltk
import os
os.environ['JAVAHOME'] =  "C:/Program Files/Java/jdk-17.0.4.1/bin/java.exe"
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.chunk import conlltags2tree
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger



text = open("./pos_reference.txt").read()
text = text.split("\n")
newText = []
tagsRef = []
for t in text:
    t = t.split("\t")
    newText.append(t[0])
    if(len(t)>1):
        tagsRef.append(t[1])

string = ""


for t in newText:
    if(t!=''):
        string = string + " " + t
    else:
        string = string + "\n"

with open("pos_text.txt",'w') as f:
    f.write(string)

##On cree une grille qui passe de tag REF à des tags PTB
tagRefPtb = open('./POSTags_REF_PTB.txt').read()
tagRefPtb = tagRefPtb.split("\n")
for i in range(len(tagRefPtb)):
    tagRefPtb[i] = tagRefPtb[i].split()


##On cree une grille qui passe de tag PTB à des tags Universal
tagPtbUni = open('./POSTags_PTB_Universal.txt').read()
tagPtbUni = tagPtbUni.split("\n")
for i in range(len(tagPtbUni)):
    tagPtbUni[i] = tagPtbUni[i].split()

tagsUni = ""
for tagRef in tagsRef:
    for i in range(len(tagRefPtb)):
        if(tagRefPtb[i][0] == tagRef):
            tagPTB = tagRefPtb[i][1]
    
    for i in range(len(tagPtbUni)):
        if(tagPtbUni[i][0] == tagPTB):
            tagUni = tagPtbUni[i][1]

    tagsUni = tagsUni + tagUni + "\n"

with open("pos_reference.txt.univ",'w') as f:
    f.write(tagsUni)


text = open("./pos_text.txt").read()

def process_text(raw_text):
	token_text = word_tokenize(raw_text)
	return token_text

def nltk_tagger(token_text):
	tagged_words = nltk.pos_tag(token_text)
	return(tagged_words)
tagged_words = nltk_tagger(process_text(text))

def stanford_tagger(token_text):
	st = StanfordPOSTagger("./stanford-tagger-4.2.0/stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger",
							'./stanford-tagger-4.2.0/stanford-postagger-full-2020-11-17/stanford-postagger.jar',
							encoding='utf-8')   
	ne_tagged = st.tag(token_text)
	return(ne_tagged)

def format(tagged_words):
    new_text = ""
    listTag=[]
    for tag in tagged_words:
        new_text += tag[0] + "\t" + tag[1] +"\n"
        listTag.append(tag[1])
        if(tag[0]=="."):
            new_text += "\n"
    return(new_text,listTag)

nltk_text,nltk_tag = format(nltk_tagger(process_text(text)))
stanford_text,stanford_tag = format(stanford_tagger(process_text(text)))

with open("pos_text.txt.pos.stan","w") as f:
    f.write(stanford_text)


with open("pos_text.txt.pos.nltk","w") as f:
    f.write(nltk_text)

tagNltkUni = ""
for tagRef in nltk_tag:
    for i in range(len(tagRefPtb)):
        if(tagRefPtb[i][0] == tagRef):
            tagPTB = tagRefPtb[i][1]
    
    for i in range(len(tagPtbUni)):
        if(tagPtbUni[i][0] == tagPTB):
            tagUni = tagPtbUni[i][1]

    tagNltkUni = tagNltkUni + tagUni + "\n"

tagStanfordUni = ""

for tagRef in stanford_tag:
    for i in range(len(tagRefPtb)):
        if(tagRefPtb[i][0] == tagRef):
            tagPTB = tagRefPtb[i][1]
    
    for i in range(len(tagPtbUni)):
        if(tagPtbUni[i][0] == tagPTB):
            tagUni = tagPtbUni[i][1]

    tagStanfordUni = tagStanfordUni + tagUni + "\n"


with open("pos_test.txt.pos.stan.univ",'w') as f:
    f.write(tagStanfordUni)

with open("pos_test.txt.pos.nltk.univ",'w') as f:
    f.write(tagNltkUni)