def get_sets(words_set, mode):
    good = set()
    bad = set()
    very_bad = set()
    
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
            if max(values) > 80.0:
                probably_good.add(word)

    return good, probably_good

def get_precision(words_set, mode, good=None, probably_good=None):
    if good is None or probably_good is None:
        good, probably_good = get_sets(words_set, mode)
    precision = ((len(good) + len(probably_good)) / len(words_set)) * 100.0
    return precision, good, probably_good

def get_recall(prdr_good_set, ap_good_set):
    if prdr_good_set is None or ap_good_set is None:
        return None
    prdr_recall = (len(prdr_good_set))/(len(prdr_good_set.union(ap_good_set))) * 100.0
    ap_recall = (len(prdr_good_set))/(len(prdr_good_set.union(ap_good_set))) * 100.0
    return prdr_recall, ap_recall