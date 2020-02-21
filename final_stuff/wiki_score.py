import wikipedia
import wikipediaapi

# wiki_wiki = wikipediaapi.Wikipedia('en') # generating wiki_wiki

# Obtaining the cs_categories set:
# cs_categories = set()
# with open('../final_stuff/data/wikipedia_reference/cs_categories.txt', 'r') as f:
#     for line in f:
#         cs_categories.add(line[:-1])

def get_wiki_score(candidate, wiki_wiki, cs_categories, num_results=20):
    relevant_pages = set()
    for suggested_page in wikipedia.search(candidate, results=num_results, suggestion=False):
        for word in wiki_wiki.page(suggested_page).categories:
            word = word[9:]
            if word in cs_categories:
                relevant_pages.add(suggested_page)
                break
    return (len(relevant_pages)/num_results)