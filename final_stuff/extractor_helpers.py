import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from spacy.tokenizer import Tokenizer
import re

from whoosh.qparser import QueryParser
import whoosh.index as index
from whoosh.index import open_dir
from whoosh.query import Every

from collections import defaultdict

import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

nlp = spacy.load('en_core_web_sm')
infixes = tuple([r"'s\b", r"(?<!\d)\.(?!\d)"]) +  nlp.Defaults.prefixes
infix_re = spacy.util.compile_infix_regex(infixes)

def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)

nlp.tokenizer = custom_tokenizer(nlp)

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
        data = file.read(size)
        if not data:
            break
        yield data
        

def getPhrases(file, context_pattern):
    new_phrases = set()
    with open(file, 'r') as f:
        matcher = Matcher(nlp.vocab)
        file_chunk = partition(f)
        for t in file_chunk:
            doc = nlp(t)
            for cp in context_pattern:
                offset = 0
                for i in range(len(cp)):
                    if 'POS' in cp[i]:
                        break
                    offset += 1
                matcher.add("extraction", None, cp)
                matches = matcher(doc)
                for match_id, start, end in matches:
                    span = doc[start+offset:end].text
                    if span not in new_phrases:
                        new_phrases.add(span)
    #                     print(span)
                matcher.remove("extraction")
    return new_phrases