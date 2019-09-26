import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

nlp = spacy.load('en_core_web_sm')

def extractor(text, pattern):
    '''
    @param str text: text source for tuple extraction
    @param list pattern: extraction pattern
    A phrase extractor based on spacy
    '''
    matcher = Matcher(nlp.vocab)
    matcher.add("test", None, pattern)
    doc = nlp(text)
    matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        print(match_id, string_id, start, end, span.text)

def main():
    pattern = [{"POS": "NOUN"}, {"POS": "NOUN"}]
    with open("test.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line)
            extractor(line, pattern)

if __name__ == '__main__':
    main()