#%%
# natural language processing module (Extractor)
import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

# search engine
from whoosh.qparser import QueryParser
import whoosh.index as index
from whoosh.index import open_dir
from whoosh.query import Every

# library for list manipulation
import itertools

# library for context graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

nlp = spacy.load('en_core_web_sm')

def extractor(document, *pattern):
    '''
    @param str file: text file for tuple extraction
    @param list pattern: extraction pattern
    A phrase extractor based on spacy
    '''
    phrases = []
    matcher = Matcher(nlp.vocab)
    matcher.add("pattern extraction", None, *pattern)
    doc = nlp(document)
    matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        phrases.append(span.text)
    return phrases

def partition(file, size = 1000000):
    '''
    partition the input file into block with maximum size of 1000000, since SpaCy v2.x parser may have issues allocating memory with size larger than 1000000
    '''
    while True:
        data = file.read(size).lower()
        if not data:
            break
        yield data

#%%
def getPhrases(file, context_pattern):
    new_phrases = set()
    with open(file, 'r') as f:
        t = f.read().lower()
        matcher = Matcher(nlp.vocab)
        doc = nlp(t)
        for cp in context_pattern:
            matcher.add("extraction", None, cp)
            matches = matcher(doc)
            for match_id, start, end in matches:
                span = doc[start+2:end].text
                if span not in new_phrases:
                    new_phrases.add(span)
            matcher.remove("extraction")
    return new_phrases

def patternSearch(T_0, file):
    phrase_patterns = set()
    seed_pattern = [nlp(x) for x in T_0]
    phrase_matcher = PhraseMatcher(nlp.vocab)
    phrase_matcher.add('pattern search', None, *seed_pattern)
    # find occurrences of seed phrases
    with open(file, "r") as f:
        document = nlp(f.read().lower())
        matches = phrase_matcher(document)
        for match_id, start, end in matches:
            p = tuple((start, end))
            if p not in phrase_patterns:
                phrase_patterns.add(p)
    # find patterns around seed phrases
    unranked_patterns = []
    with open(file, "r") as f:
        text = nlp(f.read().lower())
        for phrase_pattern in phrase_patterns:
            start = phrase_pattern[0]
            end = phrase_pattern[1]
            if (text[start - 1].text == '\n'):
                continue
            # add context pattern 
            tmp = []
            for i in range(2, 0, -1):
                tmp.append({"TEXT": text[start - i].text})
            # add content pattern 
            span = text[start:end]
            for token in span:
                tmp.append({"POS": token.pos_})
            if tmp not in unranked_patterns:
                unranked_patterns.append(tmp)
                print(tmp)
    unranked_phrases = list(getPhrases(file, unranked_patterns))
    # build context graph
    context_graph = nx.Graph()
    # add tuples and patterns into graph
    for i in range(len(unranked_phrases)):
        node = 't' + str(i)
        context_graph.add_node(node, pos=(0, i))
    for i in range(len(unranked_patterns)):
        node = 'p' + str(i)
        context_graph.add_node(node, pos=(2, i))
    context_matrix = np.zeros((len(unranked_phrases), len(unranked_patterns)))
    # find c (t, p)
    with open(file, 'r') as f:
        t = f.read().lower()
        matcher = Matcher(nlp.vocab)
        doc = nlp(t)
        for i in range(len(unranked_patterns)):
            matcher.add("extraction", None, unranked_patterns[i])
            matches = matcher(doc)
            for match_id, start, end in matches:
                span = doc[start+2:end].text
                j = unranked_phrases.index(span)
                context_matrix[j, i] += 1
            matcher.remove("extraction")
    # add context nodes into graph
    c_count = 0
    for i in range(context_matrix.shape[0]):
        for j in range(context_matrix.shape[1]):
            if context_matrix[i, j] != 0:
                occur = context_matrix[i, j]
                node_t = 't' + str(i)
                node_p = 'p' + str(j)
                node_c = 'c' + str(c_count)
                c_count += 1
                context_graph.add_node(node_c, pos=(1, c_count))
                context_graph.add_edge(node_t, node_c, weight=occur)
                context_graph.add_edge(node_c, node_p, weight=occur)
    # draw context graph
    plt.figure()
    pos=nx.get_node_attributes(context_graph,'pos')
    nx.draw(context_graph, pos, with_labels=True)
    labels = nx.get_edge_attributes(context_graph, 'weight')
    nx.draw_networkx_edge_labels(context_graph,pos,edge_labels=labels)
    # return patterns
    return unranked_phrases

#%%
seed = set(['multimedia data types', 'database system', 'cryptographic algorithm'])
new_phrases = patternSearch(seed, 'small.txt')
new_new_phrases = patternSearch(new_phrases, 'small.txt')
# print(context_p)
print(new_phrases)
print(new_new_phrases)

#%% 
seed = set(['multimedia data types', 'database system', 'cryptographic algorithm'])
new_phrases = patternSearch(seed, 'small.txt')
new_new_phrases = patternSearch(new_phrases, 'small.txt')
# print(context_p)
print(new_phrases)
print(new_new_phrases)

# %%
