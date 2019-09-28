import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

nlp = spacy.load('en_core_web_sm')

def extractor(file, *pattern):
    '''
    @param str file: text file for tuple extraction
    @param list pattern: extraction pattern
    A phrase extractor based on spacy
    '''
    phrases = []
    matcher = Matcher(nlp.vocab)
    matcher.add("pattern extraction", None, *pattern)
    with open(file, "r") as f:
        text = f.read().lower()
        doc = nlp(text)
        matches = matcher(doc)
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]  # Get string representation
            span = doc[start:end]  # The matched span
            print(match_id, string_id, start, end, span.text)
            phrases.append(span.text)
        return phrases

def main():
    # pattern1 = [{"POS": "NOUN"}, {"POS": "NOUN"}]
    pattern2 = [{"POS": "ADJ"}, {"POS": "NOUN"}, {"POS": "NOUN"}]
    # pattern = [{"POS": "NOUN"}, {"LOWER": "learning"}]
    # pattern = [{"POS": "ADJ"}, {"LOWER": "learning"}]
    p = extractor("test.txt", pattern2)
    print(p)

if __name__ == '__main__':
    main()