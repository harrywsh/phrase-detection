import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

from whoosh.qparser import QueryParser
import whoosh.index as index
from whoosh.index import open_dir
from whoosh.query import Every

import wikipedia
import wikipediaapi
import math
import time

from collections import defaultdict

import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
from os import path, remove
import copy
import pickle
from requests import ConnectTimeout, HTTPError, ReadTimeout, Timeout, ConnectionError

from prdualrank import prDualRank
from extractor_helpers import *
from wiki_score import *
# from wiki_ir_compute import * # New Addition

import re
import nltk
from nltk.corpus import stopwords
from collections import defaultdict

ngram_prob_map = []
phrase_seg_score = {}
removed_phrases = set()
wiki_score_cache = {}
# wiki_ir_cache = {}
error_count = 0

total_ngram_counts = []

final_patterns = []
final_keywords = []
pattern_to_score_map = {}
keyword_to_score_map = {}

def get_seg_score(candidate_phrase):
    global final_patterns, final_keywords, pattern_to_score_map, keyword_to_score_map, ngram_prob_map, phrase_seg_score, removed_phrases, wiki_score_cache, error_count, total_ngram_counts
    if candidate_phrase in set(phrase_seg_score.keys()).difference(removed_phrases):
        return phrase_seg_score[candidate_phrase]
    return 0.009633215 # 1/4 of avg seg_score in the file

def run_prdualrank(T_0, unranked_patterns, unranked_phrases, file):
    global final_patterns, final_keywords, pattern_to_score_map, keyword_to_score_map, ngram_prob_map, phrase_seg_score, removed_phrases, wiki_score_cache, error_count, total_ngram_counts
    
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

    with open('../development_ipynbs/context_matrix.pickle', 'rb') as f:
        context_matrix = pickle.load(f)
        print("[LOG] Loaded the context matrix. Shape: " + str(context_matrix.shape))

    for i in range(len(unranked_patterns)):
        for j in range(len(unranked_phrases)):
            if context_matrix[j, i] > 0:
                id2patterns[j].add(i)
                pattern2ids[i].add(j)
    
    id2sup = {}
    for i in range(len(unranked_phrases)):
        id2sup[i] = 0

    pattern2sup = {}
    for i in range(len(unranked_patterns)):
        pattern2sup[i] = 0

    for id in id2patterns.keys():
        sum = 0
        for col in range(len(unranked_patterns)):
            sum += context_matrix[id, col]
        id2sup[id] = sum

    for pattern in pattern2ids.keys():
        sum = 0
        for row in range(len(unranked_phrases)):
            sum += context_matrix[row, pattern]
        pattern2sup[pattern] = sum

    print("[LOG] Initiating PR Dual Rank inference.")
    l1, l2, l3, l4, m1, m2, m3, m4 = prDualRank(seedIdwConfidence, [], id2patterns, pattern2ids, {},
             {}, {}, {}, id2phrase, context_matrix.tolist(), id2sup, pattern2sup,
             FLAGS_VERBOSE=True, FLAGS_DEBUG=True)
    print("[LOG] Ended PR Dual Rank inference.")

    return l1, l2, l3, l4, m1, m2, m3, m4

def execute_ranking(T_0, file, scoring_mode, wiki_wiki, cs_categories):
    global final_patterns, final_keywords, pattern_to_score_map, keyword_to_score_map, ngram_prob_map, phrase_seg_score, removed_phrases, wiki_score_cache, error_count, total_ngram_counts

    with open('../development_ipynbs/cm_patterns.pickle', 'rb') as f:
        unranked_patterns = pickle.load(f)
        print("[LOG] Loaded all patterns. Length: " + str(len(unranked_patterns)))
    with open('../development_ipynbs/cm_words.pickle', 'rb') as f:
        unranked_phrases = pickle.load(f)
        print("[LOG] Loaded all phrases. Length: " + str(len(unranked_phrases)))

    l1, l2, l3, l4, m1, m2, m3, m4 = run_prdualrank(T_0, unranked_patterns, unranked_phrases, file)

    pattern_precision = m1
    pattern_recall = m2
    tuple_precision = m3
    tuple_recall = m4

    print("length " + str(len(pattern_precision)))

    pattern2fscore = {}
    for i in range(len(unranked_patterns)):
        precision = pattern_precision[i]
        recall = pattern_recall[i]

        f1 = 0.0
        if (recall + precision) != 0.0:
            f1 = ((2 * recall * precision) / (recall + precision))

        pattern2fscore[i] = f1

    sorted_patterns_ids = sorted(pattern2fscore, key=pattern2fscore.__getitem__, reverse=True)
    ranked_patterns = [(unranked_patterns[i], pattern2fscore[i]) for i in sorted_patterns_ids] # All patterns are now sorted!

    phrase2fscore = {}
    phrase2precision = {}
    phrase2recall = {}

    for i in range(len(unranked_phrases)):
        precision = tuple_precision[i]
        recall = tuple_recall[i]

        phrase2precision[unranked_phrases[i]] = precision
        phrase2recall[unranked_phrases[i]] = recall

        fscore = 0.0

        f1 = 0.0
        if (recall + precision) != 0.0:
            f1 = ((2 * recall * precision) / (recall + precision))

        if scoring_mode == 9: # Current Best Method
            if unranked_phrases[i] not in wiki_score_cache:
                try:
                    wiki_score_cache[unranked_phrases[i]] = get_wiki_score(unranked_phrases[i], wiki_wiki, cs_categories, 40)
                except (ConnectTimeout, HTTPError, ReadTimeout, Timeout, ConnectionError):
                    wiki_score_cache[unranked_phrases[i]] = 0.5
                    error_count += 1
            fscore = 2.718 ** (wiki_score_cache[unranked_phrases[i]] * f1 * get_seg_score(unranked_phrases[i]))
        else:
            fscore = -100
        phrase2fscore[i] = fscore
        if unranked_phrases[i] in T_0:
            phrase2fscore[i] = 100
        keyword_to_score_map[unranked_phrases[i]] = fscore

    sorted_phrases_ids = sorted(phrase2fscore, key=phrase2fscore.__getitem__, reverse=True)
    ranked_keywords = [(unranked_phrases[i], phrase2fscore[i]) for i in sorted_phrases_ids] # All keywords are now ranked!

    with open('../development_ipynbs/cm_precision.pickle', 'wb') as f:
        pickle.dump(phrase2precision, f)
        print("[LOG] Saving precision values.")
    with open('../development_ipynbs/cm_recall.pickle', 'wb') as f:
        pickle.dump(phrase2recall, f)
        print("[LOG] Saving recall values.")

    return ranked_patterns, ranked_keywords

if (__name__ == "__main__"):
    seed = set(["machine learning", "artificial intelligence", "constraint programming", "natural language processing", "databases"])# "distributed database systems"])
    final_keywords = list(copy.deepcopy(seed))
    results_filename = "./outputs/" + sys.argv[1] # "./outputs/" + "results_small.txt"
    scoring_mode = 9

    with open('./data/wiki_score_cache.txt', 'rb') as f:
        wiki_score_cache = pickle.loads(f.read())

    wiki_wiki = None
    cs_categories = None
    wiki_wiki = wikipediaapi.Wikipedia('en') # generating wiki_wiki

    cs_categories = set() # obtaining cs_categories
    with open('./data/wikipedia_reference/cs_categories.txt', 'r') as f:
        for line in f:
            cs_categories.add(line[:-1])

    # Data regarding Phrase Segmentation Scores
    phrase_seg_score = defaultdict(float)
    with open('./data/segmentation_multi_words_0_1.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            for phrase in (re.findall(re.compile(r"<phrase>(.+?)</phrase>", re.I|re.S), line.strip())):
                phrase_seg_score[phrase] += 1

    max_seg_score = 0
    for key, val in phrase_seg_score.items():
        max_seg_score = max(max_seg_score, phrase_seg_score[key])

    for key, val in phrase_seg_score.items():
        phrase_seg_score[key] = val/max_seg_score

    for stopwd in set(stopwords.words('english')):
        for wd in phrase_seg_score.keys():
            if stopwd in wd.split(' '):
                removed_phrases.add(wd)

    with open('./data/ngram_values.txt', 'rb') as f:
        ngram_prob_map = pickle.loads(f.read())

    with open('./data/total_ngram_count.txt', 'rb') as f:
        total_ngram_counts = pickle.loads(f.read())

    with open(results_filename, "w+") as f:
        start = time.time()
        ranked_patterns, ranked_keywords = execute_ranking(seed, "placeholder", scoring_mode, wiki_wiki, cs_categories)
        end = time.time()
        f.write("PRDualRank time taken is " + str(end - start) + "\n")
        print("PRDualRank finished!")

        f.write("\nFinal Sorted Keywords:\n")
        f.write(str(ranked_keywords))

        f.write("\nFinal Sorted Patterns:\n")
        f.write(str(ranked_patterns))

        with open('./data/wiki_score_cache.txt', 'wb') as f2:
            pickle.dump(wiki_score_cache, f2)

        f.write("\n[LOG]: Error count: " + str(error_count) + "\n")