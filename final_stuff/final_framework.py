import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

from whoosh.qparser import QueryParser
import whoosh.index as index
from whoosh.index import open_dir
from whoosh.query import Every

from collections import defaultdict

import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from prdualrank import prDualRank
from extractor_helpers import *

def run_prdualrank(T_0, unranked_patterns, unranked_phrases, file):

    phrase2id = {}
    for i in range(len(unranked_phrases)):
        phrase2id[unranked_phrases[i]] = i

    id2phrase = {}
    for i in range(len(unranked_phrases)):
        id2phrase[i] = unranked_phrases[i]

    id2pattern = {}
    for i in range(len(unranked_patterns)):
        id2pattern[i] = unranked_patterns[i]

    seedIdwConfidence = {}
    for key, val in phrase2id.items():
        if key in T_0:
            seedIdwConfidence[val] = 0.0

    id2patterns = defaultdict(set)
    pattern2ids = defaultdict(set)

    context_matrix = np.zeros((len(unranked_phrases), len(unranked_patterns)))
    # find c (t, p)
    with open(file, 'r') as f:
        file_chunk = partition(f)
        matcher = Matcher(nlp.vocab)
        for t in file_chunk:
            doc = nlp(t)
            for i in range(len(unranked_patterns)):
                offset = 0
                for pattern_dict in unranked_patterns[i]:
                    if 'POS' in pattern_dict:
                        break
                    offset += 1
                matcher.add("extraction", None, unranked_patterns[i])
                matches = matcher(doc)
                for match_id, start, end in matches:
                    span = doc[start+offset:end].text
                    j = unranked_phrases.index(span)
                    context_matrix[j, i] += 1
                    id2patterns[j].add(i)
                    pattern2ids[i].add(j)
                matcher.remove("extraction")


    id2sup = {key:len(val) for key, val in id2patterns.items()}
    pattern2sup = {key:len(val) for key, val in pattern2ids.items()}

    l1, l2, l3, l4, m1, m2, m3, m4 = prDualRank(seedIdwConfidence, [], id2patterns, pattern2ids, {},
             {}, {}, {}, id2phrase, context_matrix.tolist(), id2sup, pattern2sup,
             FLAGS_VERBOSE=False, FLAGS_DEBUG=False)

    return l1, l2, l3, l4, m1, m2, m3, m4

def patternSearch(T_0, T, file):
    current_patterns = [nlp(x) for x in T]
    phrase_matcher = PhraseMatcher(nlp.vocab)
    phrase_matcher.add('pattern search', None, *current_patterns)
    unranked_patterns = []
    # find occurrences of seed phrases
    with open(file, "r") as f:
        file_chunk = partition(f)
        for document in file_chunk:
            print(len(document))
            document = nlp(document)
            phrase_patterns = set()
            matches = phrase_matcher(document)
            for match_id, start, end in matches:
                p = tuple((start, end))
                if p not in phrase_patterns:
                    phrase_patterns.add(p)
    # find patterns around seed phrases
            for phrase_pattern in phrase_patterns:
                start = phrase_pattern[0]
                end = phrase_pattern[1]
                if (document[start - 1].text == '\n'):
                    continue
                # add context pattern
                tmp = []
                for i in range(2, 0, -1):
                    if document[start - 1].tag_ == "IN":
                        tmp.append({"TEXT": document[start - 1].text})
                        break
                    tmp.append({"TEXT": document[start - i].text})
                # add content pattern
                span = document[start:end]
                for token in span:
                    tmp.append({"POS": token.pos_})
                if tmp not in unranked_patterns:
                    unranked_patterns.append(tmp)
    unranked_phrases = list(getPhrases(file, unranked_patterns))

# -------- Graph Generating Code --------
#     # build context graph
#     context_graph = nx.Graph()
#     # add tuples and patterns into graph
#     for i in range(len(unranked_phrases)):
#         node = 't' + str(i)
#         context_graph.add_node(node, pos=(0, i))
#     for i in range(len(unranked_patterns)):
#         node = 'p' + str(i)
#         context_graph.add_node(node, pos=(2, i))

#     context_matrix = np.zeros((len(unranked_phrases), len(unranked_patterns)))
#     # find c (t, p)
#     with open(file, 'r') as f:
#         t = f.read().lower()
#         matcher = Matcher(nlp.vocab)
#         doc = nlp(t)
#         for i in range(len(unranked_patterns)):
#             matcher.add("extraction", None, unranked_patterns[i])
#             matches = matcher(doc)
#             for match_id, start, end in matches:
#                 span = doc[start+2:end].text
#                 j = unranked_phrases.index(span)
#                 context_matrix[j, i] += 1
#             matcher.remove("extraction")
# -------- Graph Generating Code --------

    l1, l2, l3, l4, m1, m2, m3, m4 = run_prdualrank(T_0, unranked_patterns, unranked_phrases, file)

    expanded_pattern_pre = [unranked_patterns[i] for i in l1]
    expanded_pattern_rec = [unranked_patterns[i] for i in l2]
    expanded_eid_pre = [unranked_phrases[i] for i in l3]
    expanded_eid_rec = [unranked_phrases[i] for i in l4]

    pattern2fscore = {}
    for i in range(len(unranked_patterns)):
        recall = m2[i]
        precision = m1[i]
        if (recall + precision) == 0:
            fscore = 0
        else:
            fscore = ((2 * recall * precision) / (recall + precision))
        pattern2fscore[i] = fscore
    sorted_patterns_ids = sorted(pattern2fscore, key=pattern2fscore.__getitem__, reverse=True)
    sorted_patterns = [unranked_patterns[i] for i in sorted_patterns_ids]

# -------- Graph Generating Code --------
# add context nodes into graph

#     c_count = 0
#     for i in range(context_matrix.shape[0]):
#         for j in range(context_matrix.shape[1]):
#             if context_matrix[i, j] != 0:
#                 occur = context_matrix[i, j]
#                 node_t = 't' + str(i)
#                 node_p = 'p' + str(j)
#                 node_c = 'c' + str(c_count)
#                 c_count += 1
#                 context_graph.add_node(node_c, pos=(1, c_count))
#                 context_graph.add_edge(node_t, node_c, weight=occur)
#                 context_graph.add_edge(node_c, node_p, weight=occur)
# draw context graph
#     pos=nx.get_node_attributes(context_graph,'pos')
#     nx.draw(context_graph, pos, with_labels=True)
#     labels = nx.get_edge_attributes(context_graph, 'weight')
#     nx.draw_networkx_edge_labels(context_graph,pos,edge_labels=labels)
# # -------- Graph Generating Code --------

    return sorted_patterns

def tuple_search(T_0, sorted_patterns, file, k_depth):

    sorted_patterns = sorted_patterns[0:k_depth]
    unranked_phrases = list(getPhrases(file, sorted_patterns))

    l1, l2, l3, l4, m1, m2, m3, m4 = run_prdualrank(T_0, sorted_patterns, unranked_phrases, file)

    expanded_pattern_pre = [sorted_patterns[i] for i in l1]
    expanded_pattern_rec = [sorted_patterns[i] for i in l2]
    expanded_eid_pre = [unranked_phrases[i] for i in l3]
    expanded_eid_rec = [unranked_phrases[i] for i in l4]

    phrase2fscore = {}
    for i in range(len(unranked_phrases)):
        recall = m4[i]
        precision = m3[i]
        if (recall + precision) == 0:
            fscore = 0
        else:
            fscore = ((2 * recall * precision) / (recall + precision))
        phrase2fscore[i] = fscore
    sorted_phrases_ids = sorted(phrase2fscore, key=phrase2fscore.__getitem__, reverse=True)
    sorted_phrases = [unranked_phrases[i] for i in sorted_phrases_ids]

    sorted_phrases = sorted_phrases[0:k_depth]

    return sorted_phrases

if (__name__ == "__main__"):

    seed = set(["machine learning", "query optimization", "RSA encryption", "distributed database systems"])
    keywords = seed
    filename = "./data/" + "small.txt"
    iter_num = 5
    k_depth = 50
    results_filename = "./outputs/" + "results_small.txt"
    lower_filename = filename[:-4] + "_lower.txt"
    
    with open(lower_filename, "w+") as f:
        with open(filename, "r") as fn:
            t = fn.read().lower()
            f.write(t)

    with open(results_filename, "w+") as f:
        for i in tqdm(range(iter_num)):
            print("Iteration " + str(i+1) + "...\n")
            sorted_patterns = patternSearch(seed, keywords, lower_filename)
            f.write("\nSorted Patterns:\n")
            for pattern in sorted_patterns[0:k_depth]:
                f.write(str(pattern))
                f.write("\n")
            sorted_keywords = tuple_search(seed, sorted_patterns, lower_filename, k_depth)
            f.write("Sorted Keywords:\n")
            f.write(str(sorted_keywords))
            keywords = sorted_keywords