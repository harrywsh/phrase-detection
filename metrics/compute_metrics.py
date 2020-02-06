import wikipedia
import urllib
from urllib.error import HTTPError
import difflib
import spacy
import datetime

nlp = spacy.load("en_core_web_md")

# Mode 0 for PR Dual Rank, Mode 1 for AutoPhrase
def get_sets(words_set, mode):
    good = set()
    bad = set()
    very_bad = set()
    
    prefix = "prdr_"
    if mode == 0:
        prefix = "prdr_"
    elif mode == 1:
        prefix = "ap_"      
    
    f = open(prefix + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt", 'w+')
    if mode == 0:
        f.write("PR Dual Rank Logs:\n")
    elif mode == 1:
        f.write("Auto Phrase Logs:\n")
    
    for i, word in enumerate(words_set):
        try:
            url_suffix = (word.replace(" ", "_").replace("-", "_")).capitalize()
            url = "https://en.wikipedia.org/wiki/" + url_suffix
            status = 0
            try:
                code = urllib.request.urlopen(url).getcode()
                if code == 200:
                    good.add(word)
                    status = 1
            except HTTPError:
                if status == 0 and word[-1] == 's' and len(word.split(" ")) > 1:
                    url = url[:-1]
                    try:
                        code = urllib.request.urlopen(url).getcode()
                        if code == 200:
                            good.add(word)
                            status = 1
                    except HTTPError:
                        bad.add(word)
            print("Ran " + str(i + 1) + "... Keyword: " + word + " URL: " + url + " Status: " + str(status))
            f.write("Ran " + str(i + 1) + "... Keyword: " + word + " URL: " + url + " Status: " + str(status) + "\n")
        except:
            very_bad.add(word)
    
    probably_good = set()
    for word in words_set.difference(good):
        query = word.replace("-", " ").lower()
        query_tok = nlp(query)
        values = []
        for result in wikipedia.search(query):
            result_tok = nlp(result.lower().replace("-", " "))
            d = result_tok.similarity(query_tok) * 100 
            values.append(d)
        if len(values) != 0:
            print("Rechecking... " + "Keyword: " + word + " Max Similarity: " + str(max(values)))
            f.write("Rechecking... " + "Keyword: " + word + " Max Similarity: " + str(max(values)) + "\n")
            if max(values) > 80.0:
                probably_good.add(word)
    
    f.write("Good Set:\n" + str(good) + "\n")
    f.write("Probably Good Set:\n" + str(probably_good) + "\n")
    
    f.close()
    
    return good, probably_good

def get_precision(words_set, mode, good=None, probably_good=None):
    if good is None or probably_good is None:
        good, probably_good = get_sets(words_set, mode)
    precision = ((len(good) + len(probably_good)) / len(words_set)) * 100.0
    return precision, good, probably_good


