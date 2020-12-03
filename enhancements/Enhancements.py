import sys
import re
# import datefinder
import nltk
import spacy
import neuralcoref
from nltk.tokenize import word_tokenize
import argparse
import pandas as pd
from math import isnan
import re
from datetime import datetime
# from nltk.corpus import stopwords
# stop_words = stopwords.words('english')
fake_data_path = '../epfl_server/fakes.csv'


def baseline_date_adjustment(bio):
    """
    baseline function to adjust the dates in a given biography using regex
    Args:
    bio - the biography
    Rets:
    modified_bio - the biography with the dates adjusted
    """
    match = re.findall(r'\d{4}', bio)
    sentences = nltk.tokenize.sent_tokenize(bio)
    dates = []
    date_sentences = []
    for i in match:
        for sentence in sentences:
            if i in sentence:
                date_sentences.append([int(i), sentence])
    df = pd.DataFrame(date_sentences,columns=['date', 'event'])
    sorted_events = df.sort_values('date')
    old_sentences = []
    new_sentences = []
    if (list(sorted_events['date'])[-1] - sorted_events['date'][0]) > 80:
        birth_date = sorted_events['date'][1] - 20
        old_sentences.append(sorted_events['event'][0])
        new_sentences.append(sorted_events['event'][0].replace(str(sorted_events['date'][0]), str(birth_date)))
    if 'die' in df.iloc[-1]['event'] or 'dead' in df.iloc[-1]['event'] or 'death' in df.iloc[-1]['event']:
        if df.iloc[-1]['date'] < df.iloc[0]['date']:
            death_date = sorted_events.iloc[-1]['date'] + 5
            old_sentences.append(df.iloc[-1]['event'])
            new_sentences.append(df.iloc[-1]['event'].replace(str(df.iloc[-1]['date']), str(death_date)))
    modified_bio = bio[:]
    for i in range(len(old_sentences)):
        modified_bio = modified_bio.replace(old_sentences[i], new_sentences[i])
    return modified_bio

def get_root(token):
    """
    function to get the root of a Spacy tree
    Args:
    token - a node in the tree
    Rets:
    next_ - the root of the tree where the node is
    """
    current = token
    next_ = token.head
    while (next_.text != current.text):
        current = next_
        next_ = current.head
    return next_

def all_tree_nodes(root):
    """
    function to get all nodes in a given Spacy tree
    Args:
    root - root of the tree
    Rets:
    all_nodes - a list of all the nodes in the tree
    """
    if not root.children:
        return []
    all_nodes = list(root.children)
    for child in root.children:
        all_nodes.extend(all_tree_nodes(child))
    return all_nodes

def nlp_sd_date_adjustment(bio):
    """
    function to adjust the dates in a given biography using NLP
    Args:
    bio - the biography
    Rets:
    the biography with the adjusted dates (string)
    """
    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp)
    bio_doc = nlp(bio)
#     sentences_og = nltk.tokenize.sent_tokenize(example)
#     sentences_mod = nltk.tokenize.sent_tokenize(example_doc._.coref_resolved)
    main_cluster = bio_doc._.coref_clusters[0]
    main_cluster_indices = [i[0].i for i in main_cluster]
    events = []
    born_b = False
    dead_b = False
    for i in main_cluster_indices:
        subject = bio_doc[i]
        head = get_root(bio_doc[i])
        context = all_tree_nodes(head)
        for word in context:
            match = re.findall(r'\d{4}', word.text)
            if match:
                if not (head.text == 'born' and born_b):
                    if not(head.text == 'died' and dead_b):
                        events.append([subject, head, word])
                        if head.text == 'born':
                            born_b = True
                        if head.text == 'died':
                            dead_b = True
    dates = sorted([i[2] for i in events])
    born_rep = -1
    dead_rep = -1
    born = None
    dead = None
    last_date_rep = -1
    for i in events:
        if i[1].text == 'born':
            born = i[2]
        if i[1].text == 'died':
            dead = i[2]
    if born:
        if int(dates[1].text) - int(born.text) < 10 or int(dates[1].text) - int(born.text) > 20:
            born_rep = int(dates[1].text) - 20
    if born and dead:
        if int(dead.text) - born_rep > 100 or int(dead.text) - born_rep < 1:
            dead_rep = born_rep + 80
    elif int(dates[-1].text) - int(dates[0].text) > 80:
        if born_rep == -1:
            last_date_rep = int(dates[0].text) + 80
        else:
            last_date_rep = born_rep + 80
    result_doc = bio_doc
    if born_rep != -1:
        result_doc = nlp.make_doc(bio_doc[:born.i].text + f" {born_rep}" + bio_doc[born.i+1:].text)
    if dead_rep != -1:
        result_doc = nlp.make_doc(result_doc[:dead.i].text + f" {dead_rep}" + result_doc[dead.i+1:].text)
    if last_date_rep != -1:
        result_doc = nlp.make_doc(result_doc[:dates[-1].i].text + f" {last_date_rep}" + result_doc[dates[-1].i+1:].text)
    return result_doc.text

#get historical figures and clean it
historical_figures_train = pd.read_csv('historical_figures/train1.csv')
historical_figures_test = pd.read_csv('historical_figures/test.csv')
historical_figures = pd.concat([historical_figures_train, historical_figures_test], axis=0)
historical_figures = historical_figures[historical_figures['birth_year'] != 'Unknown']
historical_figures['birth_year'] = historical_figures['birth_year'].apply(lambda i: str(i).replace('?', ''))


def get_person_replacement(person, person_info, minimum, maximum):
    """
    function to get the replacement of a person that does not fit in a biography
    Args:
    person - the Spacy token or Span object for the person to be replaced
    person_info - the information asssociated to the person as found in the historical figures dataset
    minimum - the minimum date in the biography
    maximum - the maximum date in the biography (taken as max date - 20)
    Rets:
    ret - a dictionnary containing the Spacy object as a key and its replacement as a string 
    """
    historical_figures_era = historical_figures[historical_figures['birth_year'].astype(int) > minimum]
    historical_figures_era = historical_figures_era[historical_figures_era['birth_year'].astype(int) < maximum] 

    continent = person_info['continent'].values[0]
    country = person_info['country'].values[0]
    state = person_info['state'].values[0]
    city =  person_info['city'].values[0]

    occupation = person_info['occupation'].values[0]
    industry = person_info['industry'].values[0]
    domain = person_info['domain'].values[0]

    if isinstance(city, str):
        df_geo = historical_figures_era[historical_figures_era['city'] == city]
    elif isinstance(state, str):
        df_geo = historical_figures_era[historical_figures_era['state'] == state]
    elif isinstance(country, str):
        df_geo = historical_figures_era[historical_figures_era['country'] == country]
    elif isinstance(continent, str):
        df_geo = historical_figures_era[historical_figures_era['continent'] == continent]
    else:
        df_geo = historical_figures

    df_occu = pd.DataFrame()
    if isinstance(occupation, str):
        df_occu = df_geo[df_geo['occupation'] == occupation]
    if df_occu.empty:
        if isinstance(industry, str):
            df_occu = df_geo[df_geo['industry'] == industry]
    if df_occu.empty:
        if isinstance(domain, str):
            df_occu = df_geo[df_geo['domain'] == domain]
    if df_occu.empty:
        df_occu = df_geo
    if not df_occu.empty:
        ret = {person: df_occu['full_name'].values[0]}
    else:
        ret = {person: person.text}
    return ret

def get_people_replacement(bio):
    """
    function to get the replacement of people that are in a biography
    Args:
    bio - the biography
    Rets:
    people_rep - a dictionnary containing the Spacy objects as a keys and their replacements as a strings
    """
    nlp = spacy.load('en')
    bio_doc = nlp(bio)
    people = []
    for ent in bio_doc.ents:
        if ent.label_ == 'PERSON':
            people.append(ent)
    years = []
    for ent in bio_doc.ents:
        if ent.label_ == 'DATE':
            years.append(ent)
    years = [re.findall(r'\d{4}', i.text) for i in years]
    years = [int(item) for sublist in years for item in sublist]
    if years == [] or people == []:
        return {}
    maximum = max(years) - 20
    minimum = min(years)
    people_rep = {}
    for person in people:
        if person.text in historical_figures['full_name'].values:
            person_info = historical_figures[historical_figures['full_name'] == person.text]
            birth_year = int(person_info['birth_year'].values[0])
            if(birth_year > maximum or birth_year < minimum):
                people_rep.update(get_person_replacement(person, person_info, minimum, maximum))
    return people_rep

def replace_people(bio):
    """
    function to adjust entities in a biography so that they fit in the time period
    Args:
    bio - the biography
    Rets:
    new_bio - the enhanced biography
    """
    people_rep = get_people_replacement(bio)
    new_bio = bio
    for key, value in people_rep.items():
        new_bio = new_bio.replace(key.text, value)
    return new_bio

def main(bio):
    new_bio = replace_people(nlp_sd_date_adjustment(bio))
    return new_bio

def parse_args():
    parser=argparse.ArgumentParser()
    p = parser.add_mutually_exclusive_group(required=True)
    p.add_argument('-p', '--path', type=str, help='the path to a text file containing the biography')
    p.add_argument('-t', '--text', type=str, help='the string of the biography')
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = parse_args()
    if args.text == None:
        path_to_bio = args.path
        with open(path_to_bio, 'r') as file:
            bio = file.read()
    else:
        bio = args.text
    enhanced_bio = main(bio)
    with open('enhanced_bio.txt', 'w') as file:
        file.write(enhanced_bio)