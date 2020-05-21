import wikipedia
import wikipediaapi
import math

def get_weight(position, N): # tuned for N = 100
    numerator = 1 - position * 0.01
    return numerator/100
#     numerator = 1 - position * 0.005
#     return numerator/200

def get_article_precision_and_recall(page_name, wiki_wiki, cs_categories):
    categories_list = wiki_wiki.page(page_name).categories
    num_relevant = 0
    for category in categories_list:
        category = category[9:]
        if category in cs_categories:
            num_relevant += 1
    if len(categories_list) == 0:
        return (0.0, 0.0)
    return (num_relevant/len(categories_list)), (num_relevant/100)

def get_wikipedia_precision_and_recall(candidate, wiki_wiki, cs_categories, num_results=100):
    suggested_pages = wikipedia.search(candidate, results=num_results, suggestion=False)
    result_precision = 0
    result_recall = 0
    for rank, suggested_page in enumerate(suggested_pages, 1):
        page_weight = get_weight(rank, num_results)
        page_precision, page_recall = get_article_precision_and_recall(suggested_page, wiki_wiki, cs_categories)
        result_precision += page_weight * page_precision
        result_recall += page_weight * page_recall
    return result_precision, result_recall

# wiki_wiki = wikipediaapi.Wikipedia('en') # generating wiki_wiki
#
# cs_categories = set() # obtaining cs_categories
# with open('../final_stuff/data/wikipedia_reference/cs_categories.txt', 'r') as f:
#     for line in f:
#         cs_categories.add(line[:-1])
