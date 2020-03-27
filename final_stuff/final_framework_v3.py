import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

from whoosh.qparser import QueryParser
import whoosh.index as index
from whoosh.index import open_dir
from whoosh.query import Every

import wikipedia
import wikipediaapi

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

import re
import nltk
from nltk.corpus import stopwords
from collections import defaultdict

ngram_prob_map = []
phrase_seg_score = {}
removed_phrases = set()
wiki_score_cache = {}
error_count = 0

def get_phrase_probability(phrase):
    n = len(phrase.split(' '))
    return ngram_prob_map[n - 1][phrase]

def get_better_phrase(phrase):
    words = phrase.split(' ')[1:]
    short_phrase = ""
    for word in words:
        short_phrase += " " + word
    short_phrase = short_phrase[1:]
    if get_phrase_probability(short_phrase) == 0:
        return phrase
    ratio = get_phrase_probability(phrase) / get_phrase_probability(short_phrase)
    if ratio >= 1:
        return phrase
    return short_phrase

def get_pmi(phrase):
    words = phrase.split(' ')
    ind_prob = 1.0
    for word in words:
        ind_prob *= get_phrase_probability(word)
    if ind_prob == 0:
        return 0.0
    ratio = (get_phrase_probability(phrase) / ind_prob)
    return (ratio / 2.9e8) # to keep PMI < 1

def get_seg_score(candidate_phrase):
    if candidate_phrase in set(phrase_seg_score.keys()).difference(removed_phrases):
        return phrase_seg_score[candidate_phrase]
    return 0.009633215 # 1/4 of avg seg_score in the file

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


    id2sup = {}
    pattern2sup = {}

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

    l1, l2, l3, l4, m1, m2, m3, m4 = prDualRank(seedIdwConfidence, [], id2patterns, pattern2ids, {},
             {}, {}, {}, id2phrase, context_matrix.tolist(), id2sup, pattern2sup,
             FLAGS_VERBOSE=False, FLAGS_DEBUG=False)

    return l1, l2, l3, l4, m1, m2, m3, m4

def patternSearch(T_0, T, file, scoring_mode):
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

    l1, l2, l3, l4, m1, m2, m3, m4 = run_prdualrank(T_0, unranked_patterns, unranked_phrases, file)

    expanded_pattern_pre = [unranked_patterns[i] for i in l1]
    expanded_pattern_rec = [unranked_patterns[i] for i in l2]
    expanded_eid_pre = [unranked_phrases[i] for i in l3]
    expanded_eid_rec = [unranked_phrases[i] for i in l4]

    pattern2fscore = {}
    for i in range(len(unranked_patterns)):
        recall = m2[i]
        precision = m1[i]
        fscore = 0
        if scoring_mode == 0:
            if (recall + precision) == 0:
                fscore = 0
            else:
                fscore = ((2 * recall * precision) / (recall + precision))
        elif scoring_mode == 1:
            fscore = precision
        elif scoring_mode == 2:
            fscore = recall
        elif scoring_mode == 3:
            fscore = precision * recall
        elif scoring_mode == 4:
            fscore = precision + recall
        else:
            fscore = -100
        pattern2fscore[i] = fscore
    sorted_patterns_ids = sorted(pattern2fscore, key=pattern2fscore.__getitem__, reverse=True)
    sorted_patterns = [unranked_patterns[i] for i in sorted_patterns_ids]

    return sorted_patterns

def tuple_search(T_0, sorted_patterns, file, k_depth_patterns, k_depth_keywords, scoring_mode, wiki_wiki, cs_categories):

    global wiki_score_cache

    sorted_patterns = sorted_patterns[0:k_depth_patterns]
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
        fscore = 0.0
        f1 = 0.0
        if (recall + precision) != 0.0:
            f1 = ((2 * recall * precision) / (recall + precision))
        if scoring_mode == 0:
            fscore = f1
        elif scoring_mode == 1:
            fscore = precision
        elif scoring_mode == 2:
            fscore = recall
        elif scoring_mode == 3:
            fscore = precision * recall
        elif scoring_mode == 4:
            fscore = precision + recall
        elif scoring_mode == 5:
            if unranked_phrases[i] not in wiki_score_cache:
                wiki_score_cache[unranked_phrases[i]] = get_wiki_score(unranked_phrases[i], wiki_wiki, cs_categories, 20)
            fscore = wiki_score_cache[unranked_phrases[i]]
        elif scoring_mode == 6:
            if unranked_phrases[i] not in wiki_score_cache:
                wiki_score_cache[unranked_phrases[i]] = get_wiki_score(unranked_phrases[i], wiki_wiki, cs_categories, 20)
            fscore = wiki_score_cache[unranked_phrases[i]] + f1
        elif scoring_mode == 7:
            if unranked_phrases[i] not in wiki_score_cache:
                wiki_score_cache[unranked_phrases[i]] = get_wiki_score(unranked_phrases[i], wiki_wiki, cs_categories, 20)
            fscore = wiki_score_cache[unranked_phrases[i]] + recall
        elif scoring_mode == 8:
            if unranked_phrases[i] not in wiki_score_cache:
                wiki_score_cache[unranked_phrases[i]] = get_wiki_score(unranked_phrases[i], wiki_wiki, cs_categories, 20)
            fscore = 2.718 ** (wiki_score_cache[unranked_phrases[i]]*f1)
        elif scoring_mode == 9: # Current Best Method
            if unranked_phrases[i] not in wiki_score_cache:
                try:
                    wiki_score_cache[unranked_phrases[i]] = get_wiki_score(unranked_phrases[i], wiki_wiki, cs_categories, 40)
                except (ConnectTimeout, HTTPError, ReadTimeout, Timeout, ConnectionError):
                    wiki_score_cache[unranked_phrases[i]] = 0.5
                    error_count += 1
            fscore = 2.718 ** (wiki_score_cache[unranked_phrases[i]]*f1* get_seg_score(unranked_phrases[i]))
        elif scoring_mode == 10:
            fscore = 2.718 ** (f1*get_seg_score(unranked_phrases[i]))
        elif scoring_mode == 11:
            if unranked_phrases[i] not in wiki_score_cache:
                try:
                    wiki_score_cache[unranked_phrases[i]] = get_wiki_score(unranked_phrases[i], wiki_wiki, cs_categories, 40)
                except (ConnectTimeout, HTTPError, ReadTimeout, Timeout, ConnectionError):
                    wiki_score_cache[unranked_phrases[i]] = 0.5
                    error_count += 1
            fscore = 2.718 ** (wiki_score_cache[unranked_phrases[i]] * f1 * get_seg_score(unranked_phrases[i]) * get_pmi(unranked_phrases[i]))
        elif scoring_mode == 12:
            better_phrase = get_better_phrase(unranked_phrases[i])
            if len(unranked_phrases[i].split(' ')) >= 3 and better_phrase != unranked_phrases[i]:
                unranked_phrases[i] = better_phrase
            if unranked_phrases[i] not in wiki_score_cache:
                try:
                    wiki_score_cache[unranked_phrases[i]] = get_wiki_score(unranked_phrases[i], wiki_wiki, cs_categories, 40)
                except (ConnectTimeout, HTTPError, ReadTimeout, Timeout, ConnectionError):
                    wiki_score_cache[unranked_phrases[i]] = 0.5
                    error_count += 1
            fscore = 2.718 ** (wiki_score_cache[unranked_phrases[i]]*f1* get_seg_score(unranked_phrases[i]))
        else:
            fscore = -100
        phrase2fscore[i] = fscore
    sorted_phrases_ids = sorted(phrase2fscore, key=phrase2fscore.__getitem__, reverse=True)
    sorted_phrases = [unranked_phrases[i] for i in sorted_phrases_ids]

    keyword2fscore = {}
    for i in sorted_phrases_ids:
        keyword2fscore[unranked_phrases[i]] = phrase2fscore[i]

    sorted_phrases = sorted_phrases[0:k_depth_keywords]

    return sorted_phrases, keyword2fscore

if (__name__ == "__main__"):

    seed = set(["machine learning", "artificial intelligence", "constraint programming", "natural language processing", "distributed database systems"])
    keywords = copy.deepcopy(seed)
    filename = "./data/" + sys.argv[1] # "./data/" + "small.txt"
    iter_num = 3
    k_depth_patterns = int(sys.argv[2]) # 100
    k_depth_keywords = int(sys.argv[3]) # 500
    results_filename = "./outputs/" + sys.argv[4] # "./outputs/" + "results_small.txt"
    scoring_mode = int(sys.argv[5]) # int(sys.argv[5]) # 0 or 1 or 2 or 3 or 4

    if (path.exists(filename) == False):
        print("\nWarning: the data file does not exist!\n")
        sys.exit()
    if (path.exists(results_filename) == True):
        print("\nWarning: the results file already exists! Do you really want to overwrite?\n")
        sys.exit()
    if (scoring_mode < 0 or scoring_mode > 11):
        print("\nScoring Mode is incorrect! Please retry.\n")
        sys.exit()

    print("\n[LOG]: Started run for scoring method " + str(scoring_mode) + "\n")

    lower_filename = filename[:-4] + "_lower.txt"

    keyword2fscore = {}
    for seed_word in seed:
        keyword2fscore[seed_word] = 100

    with open(lower_filename, "w+") as f:
        with open(filename, "r") as fn:
            t = fn.read().lower()
            f.write(t)

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

    with open('./data/prdr_wiki_cache.txt', 'rb') as f:
        wiki_score_cache = pickle.loads(f.read())

    with open('./data/ngram_values.txt', 'rb') as f:
        ngram_prob_map = pickle.loads(f.read())

    with open(results_filename, "w+") as f:
        for i in tqdm(range(iter_num)):
            print("Iteration " + str(i+1) + "...\n")
            sorted_patterns = patternSearch(seed, keywords, lower_filename, 0)
            f.write("\nSorted Patterns:\n")
            for pattern in sorted_patterns[0:k_depth_patterns]:
                f.write(str(pattern))
                f.write("\n")
            sorted_keywords, iter_keyword2fscore = tuple_search(seed, sorted_patterns, lower_filename, k_depth_patterns, k_depth_keywords, scoring_mode, wiki_wiki, cs_categories)
            f.write("Sorted Keywords:\n")
            f.write(str(sorted_keywords))
            keywords = keywords.union(sorted_keywords)
            for iter_word in sorted_keywords:
                if iter_word not in keyword2fscore or keyword2fscore[iter_word] < iter_keyword2fscore[iter_word]:
                    keyword2fscore[iter_word] = iter_keyword2fscore[iter_word]
        final_keywords_set = sorted(keyword2fscore, key=keyword2fscore.__getitem__, reverse=True)
        final_keywords_list = [(word, keyword2fscore[word]) for word in final_keywords_set]
        f.write("Final Sorted Keywords:\n")
        f.write(str(final_keywords_list))
        f.write("\n[LOG]: Error count: " + str(error_count) + "\n")

    remove(lower_filename)
    print("\n[LOG]: Error count: " + str(error_count) + "\n")
    print("\n[LOG]: Ended run for scoring method +" + str(scoring_mode) + "\n")
