import nltk
import os

# Configuration de l'environnement pour utiliser Java avec les taggers de Stanford
os.environ['JAVAHOME'] = "C:/Program Files/Java/jdk-17/bin/java.exe"

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.chunk import conlltags2tree
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger

# Lecture et préparation du texte à partir d'un fichier
text = open("../doc/pos_reference.txt").read()
text = text.split("\n")
newText = []
tagsRef = []

# Séparation des mots et des étiquettes de référence, si disponibles
for t in text:
    t = t.split("\t")
    newText.append(t[0])
    if (len(t) > 1):
        tagsRef.append(t[1])

string = ""

# Reconstruction du texte à partir des mots, ajout de sauts de ligne pour séparer les phrases
for t in newText:
    if (t != ''):
        string = string + " " + t
    else:
        string = string + "\n"

# Écriture du texte reconstruit dans un nouveau fichier
with open("../doc/pos_text.txt", 'w') as f:
    f.write(string)

# Lecture et création des grilles de conversion des tags de REF à PTB et de PTB à Universal
tagRefPtb = open('../doc/POSTags_REF_PTB.txt').read()
tagRefPtb = tagRefPtb.split("\n")
for i in range(len(tagRefPtb)):
    tagRefPtb[i] = tagRefPtb[i].split()

tagPtbUni = open('../doc/POSTags_PTB_Universal.txt').read()
tagPtbUni = tagPtbUni.split("\n")
for i in range(len(tagPtbUni)):
    tagPtbUni[i] = tagPtbUni[i].split()

tagsUni = ""

# Conversion des tags REF en tags Universal en utilisant les grilles de conversion
for tagRef in tagsRef:
    for i in range(len(tagRefPtb)):
        if (tagRefPtb[i][0] == tagRef):
            tagPTB = tagRefPtb[i][1]

    for i in range(len(tagPtbUni)):
        if (tagPtbUni[i][0] == tagPTB):
            tagUni = tagPtbUni[i][1]

    tagsUni = tagsUni + tagUni + "\n"

# Écriture des tags convertis en Universal dans un fichier
with open("../doc/pos_reference.txt.univ", 'w') as f:
    f.write(tagsUni)

# Ré-ouverture du texte pour étiquetage
text = open("../doc/pos_text.txt").read()


# Fonction pour tokeniser le texte
def process_text(raw_text):
    token_text = word_tokenize(raw_text)
    return token_text


# Fonction pour appliquer l'étiquetage grammatical avec NLTK
def nltk_tagger(token_text):
    tagged_words = nltk.pos_tag(token_text)
    return (tagged_words)


# Application de l'étiquetage NLTK sur le texte tokenisé
tagged_words = nltk_tagger(process_text(text))


# Fonction pour appliquer l'étiquetage grammatical avec le Stanford POS Tagger
def stanford_tagger(token_text):
    st = StanfordPOSTagger(
        "../stanford-tagger-4.2.0/stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger",
        '../stanford-tagger-4.2.0/stanford-postagger-full-2020-11-17/stanford-postagger.jar',
        encoding='utf-8')
    ne_tagged = st.tag(token_text)
    return (ne_tagged)


# Fonction pour formater les mots étiquetés en texte lisible
def format(tagged_words):
    new_text = ""
    listTag = []
    for tag in tagged_words:
        new_text += tag[0] + "\t" + tag[1] + "\n"
        listTag.append(tag[1])
        if (tag[0] == "."):
            new_text += "\n"
    return (new_text, listTag)


# Formatage et écriture des résultats des taggers NLTK et Stanford
nltk_text, nltk_tag = format(nltk_tagger(process_text(text)))
stanford_text, stanford_tag = format(stanford_tagger(process_text(text)))

with open("../doc/pos_text.txt.pos.stan", "w") as f:
    f.write(stanford_text)

with open("../doc/pos_text.txt.pos.nltk", "w") as f:
    f.write(nltk_text)

# Conversion des tags NLTK et Stanford en tags Universal et écriture dans des fichiers
tagNltkUni = ""
for tagRef in nltk_tag:
    for i in range(len(tagRefPtb)):
        if (tagRefPtb[i][0] == tagRef):
            tagPTB = tagRefPtb[i][1]

    for i in range(len(tagPtbUni)):
        if (tagPtbUni[i][0] == tagPTB):
            tagUni = tagPtbUni[i][1]

    tagNltkUni = tagNltkUni + tagUni + "\n"

tagStanfordUni = ""

for tagRef in stanford_tag:
    for i in range(len(tagRefPtb)):
        if (tagRefPtb[i][0] == tagRef):
            tagPTB = tagRefPtb[i][1]

    for i in range(len(tagPtbUni)):
        if (tagPtbUni[i][0] == tagPTB):
            tagUni = tagPtbUni[i][1]

    tagStanfordUni = tagStanfordUni + tagUni + "\n"

with open("../doc/pos_test.txt.pos.stan.univ", 'w') as f:
    f.write(tagStanfordUni)

with open("../doc/pos_test.txt.pos.nltk.univ", 'w') as f:
    f.write(tagNltkUni)