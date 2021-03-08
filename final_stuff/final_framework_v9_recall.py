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
                    j = unranked_phrases.index(span) if span in unranked_phrases else -1
                    if j == -1:
                        continue
                    context_matrix[j, i] += 1
                    id2patterns[j].add(i)
                    pattern2ids[i].add(j)
                matcher.remove("extraction")


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

    l1, l2, l3, l4, m1, m2, m3, m4 = prDualRank(seedIdwConfidence, [], id2patterns, pattern2ids, {},
             {}, {}, {}, id2phrase, context_matrix.tolist(), id2sup, pattern2sup,
             FLAGS_VERBOSE=False, FLAGS_DEBUG=False)

    return l1, l2, l3, l4, m1, m2, m3, m4

def get_new_patterns_and_phrases(T_0, T, file, max_patterns, max_keywords):
    global final_patterns, final_keywords, pattern_to_score_map, keyword_to_score_map, ngram_prob_map, phrase_seg_score, removed_phrases, wiki_score_cache, error_count, total_ngram_counts

    phrases = [keyword for keyword in T]
    unranked_patterns = []
    # find occurrences of seed phrases
    with open(file, "r") as f:
        file_chunk = partition(f)
        for document in file_chunk:
            print(len(document))
            document = nlp(document)
            patterns_to_process = []
            for word in document:
                children = []
                tokens = []
                for tok in word.subtree:
                    children.append(tok.text)
                    tokens.append(tok)
                possible_pattern = " ".join(children)
                candidate_patterns = [(phrase, tokens) for phrase in phrases if phrase in possible_pattern and phrase != possible_pattern]
                patterns_to_process.extend(candidate_patterns)
            for chunk in document.noun_chunks:
                possible_pattern = chunk.text
                candidate_patterns = [(phrase, [w for w in chunk]) for phrase in phrases if phrase in possible_pattern and phrase != possible_pattern]
                patterns_to_process.extend(candidate_patterns)  
            # patterns_to_process contains phrases and patterns. These have to be converted to usual format
            for phrase, raw_pattern in patterns_to_process:
                raw_pattern_text = [p.text for p in raw_pattern]
                phrase_len = len(phrase.split(" "))
                match_start = 0
                for i in range(len(raw_pattern_text)):
                    if phrase == " ".join(raw_pattern_text[i:(i+phrase_len)]):
                        match_start = i
                        break
                constructed_pattern = []
                for token in raw_pattern[:match_start]:
                    constructed_pattern.append({"TEXT": token.text})
                for token in raw_pattern[match_start:(match_start+phrase_len)]:
                    constructed_pattern.append({"POS": token.pos_})
                for token in raw_pattern[(match_start+phrase_len):]:
                    constructed_pattern.append({"TEXT": token.text})
                
                if raw_pattern[-1].pos_ == "PUNCT":
                    constructed_pattern = constructed_pattern[:-1]
                if raw_pattern[0].pos_ == "PUNCT":
                    constructed_pattern = constructed_pattern[1:]
                
                if constructed_pattern not in unranked_patterns and len(constructed_pattern) <= 10 and len(constructed_pattern) > 0:
                    unranked_patterns.append(constructed_pattern)
    unranked_phrases = list(getPhrases(file, unranked_patterns))

    # At this point, we have new unranked_patterns and unranked_phrases

    new_patterns = []
    new_pattern_count = 0
    for elem in unranked_patterns:
        if elem not in final_patterns:
            final_patterns.append(elem)
            new_patterns.append(elem)
            new_pattern_count += 1
            if new_pattern_count == max_patterns:
                break

    new_phrases = []
    new_phrase_count = 0
    for elem in unranked_phrases:
        if elem not in final_keywords:
            final_keywords.append(elem)
            new_phrases.append(elem)
            new_phrase_count += 1
            if new_phrase_count == max_keywords:
                break

    # print("New Patterns", new_patterns)
    # print("New Phrases", new_phrases)

    return new_patterns, new_phrases

def execute_ranking(T_0, file, scoring_mode, wiki_wiki, cs_categories):
    global final_patterns, final_keywords, pattern_to_score_map, keyword_to_score_map, ngram_prob_map, phrase_seg_score, removed_phrases, wiki_score_cache, error_count, total_ngram_counts

    unranked_patterns = final_patterns
    unranked_phrases = final_keywords

    l1, l2, l3, l4, m1, m2, m3, m4 = run_prdualrank(T_0, unranked_patterns, unranked_phrases, file)

    pattern_precision = m1
    pattern_recall = m2
    tuple_precision = m3
    tuple_recall = m4

#     pattern2fscore = {}
#     for i in range(len(unranked_patterns)):
#         precision = pattern_precision[i]
#         recall = pattern_recall[i]

#         f1 = 0.0
#         if (recall + precision) != 0.0:
#             f1 = ((2 * recall * precision) / (recall + precision))

#         pattern2fscore[i] = f1

#     sorted_patterns_ids = sorted(pattern2fscore, key=pattern2fscore.__getitem__, reverse=True)
#     ranked_patterns = [(unranked_patterns[i], pattern2fscore[i]) for i in sorted_patterns_ids] # All patterns are now sorted!
    ranked_patterns = unranked_patterns

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
        f1 = recall

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
    # global final_patterns, final_keywords, pattern_to_score_map, keyword_to_score_map, ngram_prob_map, phrase_seg_score, removed_phrases, wiki_score_cache, error_count, total_ngram_counts
    seed = set(["databases","algorithms","machine learning", "artificial intelligence","neural networks",'attention mechanisms',"constraint programming", "natural language processing",'principal component analysis','long range dependencies',"distributed database systems", "hierarchical deep reinforcement learning",'supervised machine learning models',"sequence to sequence learning"])
    final_keywords = list(copy.deepcopy(seed))
    filename = "./data/" + sys.argv[1] # "./data/" + "small.txt"
    # iter_num = 5
    max_patterns = int(sys.argv[2]) # 100
    max_keywords = int(sys.argv[3]) # 500
    results_filename = "./outputs/" + sys.argv[4] # "./outputs/" + "results_small.txt"
    scoring_mode = int(sys.argv[5]) # int(sys.argv[5]) # 9 or 12
    iter_num = int(sys.argv[6])

    if (path.exists(filename) == False):
        print("\nWarning: the data file does not exist!\n")
        sys.exit()
    if (path.exists(results_filename) == True):
        print("\nWarning: the results file already exists! Do you really want to overwrite?\n")
        sys.exit()
    if (scoring_mode != 9):
        print("\nScoring Mode is incorrect! Please retry.\n")
        sys.exit()

    print("\n[LOG]: Started run for scoring method " + str(scoring_mode) + "\n")

    lower_filename = filename[:-4] + "_lower.txt"

    with open(lower_filename, "w+") as f:
        with open(filename, "r") as fn:
            t = fn.read().lower()
            f.write(t)

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
        f.write("Hyperparameters are: \n")
        f.write('iter_num: ' + str(iter_num) + "\n")
        f.write('max_patterns ' + str(max_patterns) + "\n")
        f.write('max_keywords ' + str(max_keywords) + "\n")
        for i in tqdm(range(iter_num)):
            print("Iteration " + str(i+1) + "...\n")
            f.write("\n\nIteration " + str(i+1) + "...\n")
            start = time.time()
            new_patterns, new_phrases = get_new_patterns_and_phrases(seed, final_keywords, lower_filename, max_patterns, max_keywords) # <--- final_keywords, final_patterns updated here
            end = time.time()
            f.write("Iteration time taken is " + str(end - start) + "\n")
            f.write("New Patterns " + str(new_patterns) + "\n")
            f.write("New Phrases " + str(new_phrases) + "\n")
        print("Executing PRDualRank now...")
        start = time.time()
        ranked_patterns, ranked_keywords = execute_ranking(seed, lower_filename, scoring_mode, wiki_wiki, cs_categories)
        end = time.time()
        f.write("PRDualRank time taken is " + str(end - start) + "\n")
        print("PRDualRank finished!")

        # result_keywords_set = sorted(keyword_to_score_map, key=keyword_to_score_map.__getitem__, reverse=True)
        # result_keywords_list = [(word, keyword_to_score_map[word]) for word in result_keywords_set]

        f.write("\nFinal Sorted Keywords:\n")
        f.write(str(ranked_keywords))

        f.write("\nFinal Sorted Patterns:\n")
        f.write(str(ranked_patterns))

        with open('./data/wiki_score_cache.txt', 'wb') as f2:
            pickle.dump(wiki_score_cache, f2)

        f.write("\n[LOG]: Error count: " + str(error_count) + "\n")

    remove(lower_filename)
    print("\n[LOG]: Error count: " + str(error_count) + "\n")
    print("\n[LOG]: Ended run for scoring method +" + str(scoring_mode) + "\n")
