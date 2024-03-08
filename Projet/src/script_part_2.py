import os

import nltk

# Configuration nécessaire pour faire fonctionner le Stanford NER Tagger sous Windows
os.environ['JAVAHOME'] =  "C:/Program Files/Java/jdk-17/bin/java.exe"
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordNERTagger


# 1. Extraction de phrases à partir d'un fichier au format CoNLL pour la préparation des données d'entrée.
# Fonction pour convertir un fichier CoNLL en fichier texte
def conll_to_txt(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as input_file, \
         open(output_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            if line.strip():  # Si la ligne n'est pas vide
                word = line.split('\t')[0]  # Extraire le mot
                output_file.write(word + ' ')  # Écrire le mot dans le fichier de sortie
            else:
                output_file.write('\n')  # Nouvelle ligne pour les séparations de phrases


conll_to_txt('ne_reference.txt.conll', '../doc/ne_test.txt')

# 2. Application des deux reconnaisseurs d'entités nommées (NE Recognizers) sur le fichier préparé.
with open("../doc/ne_test.txt", 'r') as file:
    text = file.read()

tokenized_text = word_tokenize(text)

# Utilisation du Stanford NER Tagger pour identifier les entités nommées.
def stanford_ne_tagger(token_text):
    st = StanfordNERTagger('../stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz',
                           '../stanford-ner-2020-11-17/stanford-ner.jar',
                           encoding='utf-8')
    ne_tagged = st.tag(token_text)
    return ne_tagged

# Utilisation du NE Recognizer de NLTK pour identifier les entités nommées.
def nltk_ne_chunker(token_text):
    ne_chunked = nltk.ne_chunk(nltk.pos_tag(token_text), binary=False)
    iob_tagged = nltk.tree2conlltags(ne_chunked)
    return iob_tagged

# Application des taggers et formatage des résultats.
stanford_tagged = stanford_ne_tagger(tokenized_text)
nltk_chunked = nltk_ne_chunker(tokenized_text)

# Formatage des résultats des taggers pour une cohérence de sortie.
def format_ne_results(tagged_words):
    formatted_text = ""
    for tag in tagged_words:
        if len(tag) == 3:  # Pour les résultats de NLTK ne_chunk.
            word, pos, ne_tag = tag
        elif len(tag) == 2:  # Pour les résultats du Stanford NER.
            word, ne_tag = tag
        else:
            raise ValueError("Tagged words tuple is neither a pair nor a triplet.")

        # Formatage et ajout des mots taggés au texte final.
        if ne_tag == 'O':
            formatted_text += word + "\t" + ne_tag + "\n"
        else:
            formatted_text += word + "\t" + ne_tag + "\n"

        # Ajout d'une ligne vide après chaque phrase pour respecter le formatage CoNLL.
        if word == '.':
            formatted_text += "\n"
        elif word == '\'\'':
            formatted_text += "\n"
    return formatted_text

stanford_formatted = format_ne_results(stanford_tagged)
nltk_formatted = format_ne_results(nltk_chunked)

# Sauvegarde des résultats formatés pour les deux taggers.
with open("../doc/ne_test.txt.ne.stan", 'w') as file:
    file.write(stanford_formatted)

with open("../doc/ne_test.txt.ne.nltk", 'w') as file:
    file.write(nltk_formatted)

# 3. Conversion des résultats des taggers en format CoNLL-2003 pour une standardisation des étiquettes d'entités nommées.
def map_nltk_tag_to_conll(tag):
    """
    Conversion des étiquettes NE de NLTK vers le format CoNLL-2003.
    """
    mapping = {
        'B-PERSON': 'B-PER',
        'I-PERSON': 'I-PER',
        'B-ORGANIZATION': 'B-ORG',
        'I-ORGANIZATION': 'I-ORG',
        'B-GPE': 'B-LOC', # GPE est généralement converti en LOC dans CoNLL-2003.
        'I-GPE': 'I-LOC',
        'O': 'O'
    }
    return mapping.get(tag, 'O')

# Conversion des résultats de NLTK vers le format CoNLL.
def convert_nltk_to_conll(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as infile, open(output_filepath, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.strip():
                word, nltk_tag = line.strip().split()
                conll_tag = map_nltk_tag_to_conll(nltk_tag)
                outfile.write(f"{word}\t{conll_tag}\n")
            else:
                outfile.write('\n')  # Séparation des phrases par une ligne vide pour le format CoNLL.

convert_nltk_to_conll('../doc/ne_test.txt.ne.nltk', '../doc/ne_test.txt.ne.nltk.conll')

# Conversion des étiquettes de Stanford en format CoNLL-2003.
def map_stanford_tag_to_conll(tag):
    """
    Conversion des étiquettes NE de Stanford vers le format CoNLL-2003.
    """
    mapping = {
        'PERSON': 'PER',
        'ORGANIZATION': 'ORG',
        'LOCATION': 'LOC',
        'MISC': 'MISC',
        'O': 'O'
    }
    if tag != 'O':
        # Stanford NE Tagger ne spécifie pas B- ou I-, par défaut B- est ajouté. Une logique plus complexe est nécessaire pour une précision accrue.
        return f"B-{mapping[tag]}"
    else:
        return tag

# Conversion des résultats de Stanford vers le format CoNLL.
def convert_stanford_to_conll(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as infile, open(output_filepath, 'w', encoding='utf-8') as outfile:
        previous_tag = None
        for line in infile:
            if line.strip():
                word, stanford_tag = line.strip().split()
                conll_tag = map_stanford_tag_to_conll(stanford_tag)
                # Logique pour gérer le préfixe 'I-' en fonction du tag précédent.
                if previous_tag == stanford_tag and stanford_tag != 'O':
                    conll_tag = conll_tag.replace('B-', 'I-')
                outfile.write(f"{word}\t{conll_tag}\n")
                previous_tag = stanford_tag
            else:
                outfile.write('\n')  # Séparation des phrases par une ligne vide pour le format CoNLL.
                previous_tag = None

convert_stanford_to_conll('../doc/ne_test.txt.ne.stan', '../doc/ne_test.txt.ne.stan.conll')