#%%
import os
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser
from whoosh import index

def createIndex(file):
    schema = Schema(content=TEXT(stored=True))
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")
    ix = index.create_in("indexdir", schema)
    writer = ix.writer()
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            writer.add_document(content=line)
    writer.commit()

createIndex('test.txt')