import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

from whoosh.qparser import QueryParser
import whoosh.index as index
from whoosh.index import open_dir
from whoosh.query import Every

from collections import defaultdict

import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def computePatternRecallFromTuple(sg, total_eids, eids2Sup, eid2recall, eidsFeatureCount):
      tmp_sum = 0.0
      for eid in range(total_eids):
        if eids2Sup[eid] > 0:
          tmp_sum += eid2recall.get(eid, 0) * eidsFeatureCount[eid][sg] / eids2Sup[eid]
      return tmp_sum

def computeTupleRecallFromPattern(eid, extr_patterns, eidsFeatureCount, patterns2Sup, pattern2recall):
      tmp_sum = 0.0
      for sg in extr_patterns:
        tmp_sum += pattern2recall.get(sg, 0) * eidsFeatureCount[eid][sg] / patterns2Sup[sg]
      return tmp_sum

def doInferenceRecallExtPatternOnly(extr_patterns, pattern2recall, total_eids, eid2recall, 
                                    eids2Sup, eidsFeatureCount, patterns2Sup):
      for sg in extr_patterns:
        pattern2recall[sg] = computePatternRecallFromTuple(sg, total_eids, eids2Sup, 
                                                           eid2recall, eidsFeatureCount)
      for eid in range(total_eids):
        eid2recall[eid] = computeTupleRecallFromPattern(eid, extr_patterns, eidsFeatureCount, 
                                                        patterns2Sup, pattern2recall)


# In[46]:


def computePatternPrecisionFromTuple(p, pattern2eids, eidsFeatureCount, 
                                     pattern2precision, eid2precision, patterns2Sup):
      tmp_sum = 0.0
      for eid in pattern2eids[p]:
        sup = eidsFeatureCount[eid][p]
        tmp_sum += eid2precision[eid]*sup
      total = patterns2Sup[p]
      if total > 0:
        return tmp_sum / total
      else: return 0.0

def computeTuplePrecisionFromPattern(eid, groundTruthEids, eid2precision, 
                                     extr_patterns, eidsFeatureCount, eid2patterns, 
                                     pattern2precision, eids2Sup):
      if eid in groundTruthEids:
        return eid2precision[eid]
      tmp_sum = 0.0
      for sg in eid2patterns[eid]:
        sup = eidsFeatureCount[eid][sg]
        tmp_sum += pattern2precision[sg]*sup
      total = eids2Sup[eid]
      if total > 0:
        return tmp_sum / total
      else: return 0.0

def doInferencePrecExtPatternOnly(extr_patterns, pattern2eids, eidsFeatureCount, 
                                  pattern2precision, eid2precision, patterns2Sup, 
                                  total_eids, groundTruthEids, eid2patterns, eids2Sup):
      for sg in extr_patterns:
        pattern2precision[sg] = computePatternPrecisionFromTuple(sg, pattern2eids, eidsFeatureCount, 
                                                                 pattern2precision, eid2precision, patterns2Sup)

      for eid in range(total_eids):
        if eid not in groundTruthEids:
          eid2precision[eid] = computeTuplePrecisionFromPattern(eid, groundTruthEids, eid2precision,extr_patterns, 
                                                                eidsFeatureCount, eid2patterns, pattern2precision, 
                                                                eids2Sup)


# In[47]:


# expand the set of seedEntities and return eids by order, excluding seedEntities (original children)
def prDualRank(seedEidsWithConfidence, negativeSeedEids, eid2patterns, pattern2eids, eidAndPattern2strength,
             eid2types, type2eids, eidAndType2strength, eid2ename, eidsFeatureCount, eids2Sup, patterns2Sup, 
             FLAGS_VERBOSE=False, FLAGS_DEBUG=False):
      ''' 

      :param seedEidsWithConfidence: a list of [eid (int), confidence_score (float)]
      :param negativeSeedEids: a set of eids (int) that should not be included
      :param eid2patterns:
      :param pattern2eids:
      :param eidAndPattern2strength:
      :param eid2types:
      :param type2eids:
      :param eidAndType2strength:
      :param eid2ename:

      :return: P and T sorted by precision and recall
      '''
      # print(patterns2Sup)
      seedEids = [key for key, val in seedEidsWithConfidence.items()]
      groundTruthEids = [key for key, val in seedEidsWithConfidence.items()]
      eid2confidence = {key:val for key, val in seedEidsWithConfidence.items()}

      ## Cache the seedEids for later use
      cached_seedEids = set([ele for ele in seedEids])
      if FLAGS_VERBOSE:
        print('Seed set:')
        for eid in seedEids:
          print(eid, eid2ename[eid])
        print("[INFO] Start SetExpan")

    ## establish ground truth
    ##double but not int
      total_eids = len(eid2ename)
      eid2precision = {x: 0.0 for x in range(total_eids)}
      for eid in seedEids:
        eid2precision[eid] = 1.0
      tmp_sum = len(seedEids)
      eid2recall = {eid: eid2precision[eid]/tmp_sum for eid in seedEids}

      extr_patterns = list(pattern2eids.keys())

      pattern2precision = {sg: 0.0 for sg in extr_patterns}
      pattern2recall = {sg: 0.0 for sg in extr_patterns}


      iters = 0
      while iters < 20:
        iters += 1
        prev_seeds = set(seedEids)
        #start = time.time()

        #QuestP till convergence
        doInferencePrecExtPatternOnly(extr_patterns, pattern2eids, eidsFeatureCount, 
                                      pattern2precision, eid2precision, patterns2Sup, 
                                      total_eids, groundTruthEids, eid2patterns, eids2Sup)
        #QuestR till convergence
        doInferenceRecallExtPatternOnly(extr_patterns, pattern2recall, total_eids, 
                                        eid2recall, eids2Sup, eidsFeatureCount, patterns2Sup)


        #all_end = time.time()

        if FLAGS_DEBUG:
          print("End of iteration %s" % iters)

      expanded_eid_pre = sorted(eid2precision, key=eid2precision.__getitem__, reverse=True)
      expanded_eid_rec = sorted(eid2recall, key=eid2recall.__getitem__, reverse=True)
      expanded_pattern_pre = sorted(pattern2precision, key=pattern2precision.__getitem__, reverse=True)
      expanded_pattern_rec = sorted(pattern2recall, key=pattern2recall.__getitem__, reverse=True)

      print("prdr length " + str(len(pattern2precision)))
      return expanded_pattern_pre, expanded_pattern_rec, expanded_eid_pre, expanded_eid_rec, pattern2precision, pattern2recall, eid2precision, eid2recall