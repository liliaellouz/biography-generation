import wikipediaapi
import pandas as pd
import wikipedia
wikipedia.set_lang('en') # setting wikipedia language
import sys
from time import sleep
import re


def fetch_category_members(category_members, level=0, max_level=1):
        """
        Function to take all articles in category (max_level controls the depth of articles taken from the subcategories)
        Arguments:
        category_members - a list of category members
        level - the level at which to start getting articles
        max_level - the maximal level for the fetched articles
        Returns:
        article_names - a list of the desired article names
        """
        article_names = []
        for c in category_members.values():
            if c.ns == 0:
                article_names.append(c) 
                #print("%s: %s (ns: %d)" % ("*" * (level + 1), c.title, c.ns))
            elif level < max_level and c.ns == wikipediaapi.Namespace.CATEGORY:
                sub_list = []
                sub_list = fetch_category_members(c.categorymembers, level=level + 1, max_level=max_level)
                article_names = article_names + sub_list
        return article_names

def get_words(article_names):
        """
        Function that tokenizes and returns all words in the given list of articles 
        Arguments:
        article_names - list of articles
        Returns:
        words_df - the words in the articles in a dataframe
        """
        len_time = len(article_names)*0.05
        words_df = pd.DataFrame(columns=['biography'])
        for i in range(len(article_names)):
            try:
                page = wikipedia.page(article_names[i].title)
            except wikipedia.DisambiguationError as e:
                s = e.options
                s = list(filter(lambda x : x != "", s))
                try :
                    page = wikipedia.page(s)
                except wikipedia.DisambiguationError as e:
                    pass
            except wikipedia.PageError:
                pass
            words_df.loc[i] = [page.content]
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*int((i+1)/len_time), int(5*(i+1)/len_time)))
            sys.stdout.flush()
            sleep(0.25)
        return words_df


wiki_wiki = wikipediaapi.Wikipedia('en') # getting articles in english
# fetching the articles for categories of interest
people_pages = wiki_wiki.page("Category:People from Venice")
people_articles = fetch_category_members(people_pages.categorymembers, level = 0, max_level = 7)

venetian_biographies = get_words(people_articles)

# Clean the data by removing duplicates
venetian_biographies = venetian_biographies.drop_duplicates().reset_index(drop=True)

# Remove anything between equal signs like "== Overview =="
# Remove anything between <> signs
venetian_biographies = venetian_biographies.biography.str.replace("<(.*?)>|=+ (.*?) =+","").to_frame()

# compression_opts = dict(method='zip',
#                         archive_name='venetian_bios.csv')  
venetian_biographies.to_csv('venetian_bios.csv', index=False) 
