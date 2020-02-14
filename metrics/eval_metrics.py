import wikipedia
import urllib
from urllib.error import HTTPError
import difflib
import spacy
import datetime
import numpy as np

nlp = spacy.load("en_core_web_md")

from eval_metrics_helpers import *

'''
Function that computes precision and recall for keyword detection systems

* words_matrix: a list of lists of system keyword results (for e.g.: words_matrix = [prdr_results, ap_words], where prdr_results and ap_words are lists of the respective results
'''
def get_precision_and_recall_values(words_matrix, upper_limit=1100, step_size=100):
    
    num_lists = len(words_matrix)
    num_doc_values = np.arange(100, upper_limit, step_size)
    
    precision = [[] for i in range(num_lists)]
    recall = [[] for i in range(num_lists)]
    
    for threshold in num_doc_values:
        
        good_set = [set() for i in range(num_lists)]
        prolly_good_set = [set() for i in range(num_lists)]
        combined_results_set = set()
        
        for i in range(num_lists):
            prec, good_set[i], prolly_good_set[i] = get_precision(set(words_matrix[i][0:threshold]), 0, None, None)
            precision[i].append(prec)
            combined_results_set = combined_results_set.union(good_set[i], prolly_good_set[i])
        
        for i in range(num_lists):
            recall[i].append((100 * len(good_set[i].union(prolly_good_set[i])))/(len(combined_results_set)))

    
    return num_doc_values, precision, recall

'''
Visualization Example: (below code has been tried successfully in a Jupyter notebook)

import matplotlib.pyplot as plt
%matplotlib inline

x = y

prdr_precision = p1
ap_precision = p2
combined_precision = p3

# old_precision = [86.0, 82.0, 82.0, 81.75, 81.2]
# old_recall = [49.43, 53.07, 51.57, 51.58, 50.69]
# old_f1 = [62.77, 64.44, 63.32, 63.25, 63.25]

plt.figure(figsize=(10,10))
plt.scatter(x, prdr_precision, c='b', marker='x', label='PRDR')
plt.plot(x, prdr_precision, c='b')
plt.scatter(x, ap_precision, c='r', marker='s', label='AP')
plt.plot(x, ap_precision, c='r')
plt.scatter(x, combined_precision, c='g', marker='v', label='Combined')
plt.plot(x, combined_precision, c='g')
plt.legend(loc='upper right')

plt.xlabel('Number of (Top Ranked) Keywords', fontsize=12)
plt.ylabel('Precision %', fontsize=12)

for i_x, i_y in zip(x, prdr_precision):
    plt.text(i_x, i_y, '{}'.format(round(i_y,2)), color = 'blue')

for i_x, i_y in zip(x, ap_precision):
    plt.text(i_x, i_y, '{}'.format(round(i_y,2)), color = 'red')

for i_x, i_y in zip(x, combined_precision):
    plt.text(i_x, i_y, '{}'.format(round(i_y,2)), color = 'green')

# plt.show()
plt.savefig('combined_precision_graph.png')
'''