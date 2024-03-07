import nltk
import numpy
import os
os.environ['JAVAHOME'] =  "C:/Program Files/Java/jdk-17.0.4.1/bin/java.exe"
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.chunk import conlltags2tree
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger


# Question 1 : extraire les phrases de ne_reference.txt.conll dans ne_test.txt
def extract_sentences_from_conll(input, output):
    sentences = []
    current_sentence = []

    with open(input, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                current_sentence.append(line.split('\t')[0])
            else:
                if current_sentence:
                    sentences.append(' '.join(current_sentence) + "\n")
                    current_sentence = []
        if current_sentence:
            sentences.append(' '.join(current_sentence) + "\n")

    with open(output, 'w', encoding='utf-8') as file:
        file.writelines(sentences)

extract_sentences_from_conll('ne_reference.txt.conll', 'ne_test.txt')

# Question 2 : Lancer les deux NE recognizers sur le fichier ne_test.txt

with open("./ne_test.txt", 'r') as file:
    text = file.read()

tokenized_text = word_tokenize(text)

# Tokeniser le texte avec Stanford NER Tagger
def stanford_ne_tagger(token_text):
    st = StanfordNERTagger('./stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz',
                           './stanford-ner-2020-11-17/stanford-ner.jar',
                           encoding='utf-8')
    ne_tagged = st.tag(token_text)
    return ne_tagged

# Tokeniser le texte avec le NE Recognizer de NLTK
def nltk_ne_chunker(token_text):
    ne_chunked = nltk.ne_chunk(nltk.pos_tag(token_text), binary=False)
    iob_tagged = nltk.tree2conlltags(ne_chunked)
    return iob_tagged

# Utiliser les taggers et formatter les résultats
stanford_tagged = stanford_ne_tagger(tokenized_text)
nltk_chunked = nltk_ne_chunker(tokenized_text)


def format_ne_results(tagged_words):
    formatted_text = ""
    for tag in tagged_words:
        if len(tag) == 3:  # Si c'est un triplet (NLTK ne_chunk)
            word, pos, ne_tag = tag
        elif len(tag) == 2:  # Si c'est une paire (Stanford NER)
            word, ne_tag = tag
        else:
            raise ValueError("Tagged words tuple is neither a pair nor a triplet.")

        if ne_tag == 'O':
            formatted_text += word + "\t" + ne_tag + "\n"
        else:
            formatted_text += word + "\t" + ne_tag + "\n"

        if word == '.':
            formatted_text += "\n"
    return formatted_text

stanford_formatted = format_ne_results(stanford_tagged)
nltk_formatted = format_ne_results(nltk_chunked)

with open("ne_test.txt.ne.stan", 'w') as file:
    file.write(stanford_formatted)

with open("ne_test.txt.ne.nltk", 'w') as file:
    file.write(nltk_formatted)


# Question 3 : Convertir les résultats des deux NE Recognizers en utilisant les étiquettes CoNLL-2003

def map_nltk_tag_to_conll(tag):
    """
    Map NLTK's NE tags to CoNLL-2003 tags.
    """
    mapping = {
        'B-PERSON': 'B-PERS',
        'I-PERSON': 'I-PERS',
        'B-ORGANIZATION': 'B-ORG',
        'I-ORGANIZATION': 'I-ORG',
        'B-GPE': 'B-LOC', # GPE (Geo-Political Entity) est généralement mappé sur LOC
        'I-GPE': 'I-LOC',
        # Ajoutez d'autres correspondances si nécessaire
        'O': 'O'
    }
    return mapping.get(tag, 'O')

def convert_nltk_to_conll(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as infile, open(output_filepath, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.strip():  # Vérifier que la ligne n'est pas vide
                word, nltk_tag = line.strip().split()
                conll_tag = map_nltk_tag_to_conll(nltk_tag)
                outfile.write(f"{word}\t{conll_tag}\n")
            else:
                outfile.write('\n')  # Séparer les phrases par une ligne vide

convert_nltk_to_conll('ne_test.txt.ne.nltk', 'ne_test.txt.ne.nltk.conll')

def map_stanford_tag_to_conll(tag):
    """
    Map Stanford's NE tags to CoNLL-2003 tags.
    """
    mapping = {
        'PERSON': 'PERS',
        'ORGANIZATION': 'ORG',
        'LOCATION': 'LOC',
        'MISC': 'MISC',
        'O': 'O'
    }

    # Pour Stanford, les tags ne comprennent pas de préfixe 'B-' ou 'I-', nous devons les ajouter.
    # Pour l'instant, nous allons simplement ajouter 'B-' par défaut. Pour un mappage plus précis, une logique supplémentaire serait nécessaire.
    if tag != 'O':
        return f"B-{mapping[tag]}"
    else:
        return tag

def convert_stanford_to_conll(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as infile, open(output_filepath, 'w', encoding='utf-8') as outfile:
        previous_tag = None
        for line in infile:
            if line.strip():  # Vérifier que la ligne n'est pas vide
                word, stanford_tag = line.strip().split()
                # Mappage des tags Stanford vers les tags CoNLL
                conll_tag = map_stanford_tag_to_conll(stanford_tag)
                # Ajouter la logique pour le préfixe 'I-'
                if previous_tag == stanford_tag and stanford_tag != 'O':
                    conll_tag = conll_tag.replace('B-', 'I-')
                outfile.write(f"{word}\t{conll_tag}\n")
                previous_tag = stanford_tag
            else:
                outfile.write('\n')  # Séparer les phrases par une ligne vide
                previous_tag = None

# Exécutez cette fonction pour le fichier de Stanford NE Recognizer
convert_stanford_to_conll('ne_test.txt.ne.stan', 'ne_test.txt.ne.stan.conll')