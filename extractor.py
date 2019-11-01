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
    content_pattern = []
    context_pattern = []
    with open(file, "r") as f:
        text = nlp(f.read().lower())
        for phrase_pattern in phrase_patterns:
            start = phrase_pattern[0]
            end = phrase_pattern[1]
            if (text[start - 1].text == '\n'):
                continue
            # # add content pattern 
            tmp = []
            for i in range(2, 0, -1):
                tmp.append({"TEXT": text[start - i].text})
            # tmp.append({"TEXT": {"REGEX": "[a-zA-Z0-9_]"}, "OP":"+"})
            span = text[start:end]
            for token in span:
                tmp.append({"POS": token.pos_})
            # for i in range(3):
            #     tmp.append({"TEXT": text[end + i].text})
            #     if text[end + i].text == '\n':
            #         break
            if tmp not in context_pattern:
                context_pattern.append(tmp)
                print(tmp)

    # get new phrases
    return context_pattern

def phraseExtraction(file, context_pattern):
    new_phrases = set()
    with open('test.txt', 'r') as f:
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

#%%
seed = set(['multimedia data types', 'database system', 'cryptographic algorithm'])
context_p = patternSearch(seed, 'test.txt')
# print(context_p)
new_phrases = phraseExtraction('test.txt', context_p)
print(new_phrases)

new_p = patternSearch(new_phrases, 'test.txt')
pnn = phraseExtraction('test.txt', new_p)
print(pnn)

# %%
