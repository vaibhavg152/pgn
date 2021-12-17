from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cytoolz import concat, curry
import re
import six
import itertools
import statistics
import pickle

import collections
print('hee')

from collections import Counter
from gensim.models import KeyedVectors as kv
from rouge import Rouge 
from scipy.special import softmax
print('hee')
import nltk
# nltk.download('punkt')
print('heee')
from nltk import tokenize
from nltk import download
from nltk.corpus import stopwords
import numpy as np
# download('stopwords')
print('heeeee')



from collections import Counter
from gensim.models import KeyedVectors as kv

import nltk
#nltk.download('punkt')
from nltk import tokenize
from nltk import download
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stem = SnowballStemmer("english")
import numpy as np

# from scipy.stats import pearsonr

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
#download('stopwords')
from rouge_score import rouge_scorer


def tokenizef(text, stemmer):
  """Tokenize input text into a list of tokens.
  This approach aims to replicate the approach taken by Chin-Yew Lin in
  the original ROUGE implementation.
  Args:
    text: A text blob to tokenize.
    stemmer: An optional stemmer.
  Returns:
    A list of string tokens extracted from input text.
  """

  # Convert everything to lowercase.
  text = text.lower()
  # Replace any non-alpha-numeric characters with spaces.
  text = re.sub(r"[^a-z0-9]+", " ", six.ensure_str(text))

  tokens = re.split(r"\s+", text)
  if stemmer:
    # Only stem words more than 3 characters long.
    tokens = [stem.stem(x) if len(x) > 3 else x for x in tokens]

  # One final check to drop any empty or invalid tokens.
  tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", six.ensure_str(x))]

  return tokens


def make_n_grams(seq, n):
    """ return iterator """
    ngrams = (tuple(seq[i:i+n]) for i in range(len(seq)-n+1))
    # print(*ngrams)
    return ngrams

def _n_gram_match(summ, ref, n):
    summ_grams = Counter(make_n_grams(summ, n))
    
    ref_grams = Counter(make_n_grams(ref, n))
    grams = min(summ_grams, ref_grams, key=len)
    count = sum(min(summ_grams[g], ref_grams[g]) for g in grams)
    return count


def compute_rouge_n(output, reference, n=1, mode='f'):
    """ compute ROUGE-N for a single pair of summary and reference"""
    assert mode in list('fpr')  # F-1, precision, recall
    match = _n_gram_match(reference, output, n)
    if match == 0:
        score = 0.0
    else:
        precision = match / len(output)
        recall = match / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score


def _split_into_words(sentences):
    """Splits multiple sentences into words and flattens the result"""
    return list(itertools.chain(*[_.split() for _ in sentences]))





def _lcs_dp(a, b):
    """ compute the len dp of lcs"""
    dp = [[0 for _ in range(0, len(b)+1)]
          for _ in range(0, len(a)+1)]
    # dp[i][j]: lcs_len(a[:i], b[:j])
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp

def _lcs_len(a, b):
    """ compute the length of longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    return dp[-1][-1]


def compute_rouge_l(output, reference, mode='f'):
    """ compute ROUGE-L for a single pair of summary and reference
    output, reference are list of words
    """
    assert mode in list('fpr')  # F-1, precision, recall
    lcs = _lcs_len(output, reference)
    if lcs == 0:
        score = 0.0
    else:
        precision = lcs / len(output)
        recall = lcs / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        if mode == 'r':
            score = recall
        else:
            score = f_score
    return score


def _lcs(a, b):

    """ compute the longest common subsequence between a and b"""

    dp = _lcs_dp(a, b)
    i = len(a)
    j = len(b)
    lcs = collections.deque()
    while (i > 0 and j > 0):

        if a[i-1] == b[j-1]:
            lcs.appendleft(a[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    assert len(lcs) == dp[-1][-1]
    return lcs

def compute_rouge_l_summ(summs, refs, mode='f'):
    """ summary level ROUGE-L"""
    assert mode in list('fpr')  # F-1, precision, recall
    tot_hit = 0
    print(refs)
    ref_cnt = Counter(concat(refs))
    print(ref_cnt)
    summ_cnt = Counter(concat(summs))
    for ref in refs:
        print(ref)
        for summ in summs:
            lcs = _lcs(summ, ref)
            for gram in lcs:
                if ref_cnt[gram] > 0 and summ_cnt[gram] > 0:
                    tot_hit += 1
                ref_cnt[gram] -= 1
                summ_cnt[gram] -= 1
    if tot_hit == 0:
        score = 0.0
    else:
        precision = tot_hit / sum((len(s) for s in summs))
        recall = tot_hit / sum((len(r) for r in refs))
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        if mode == 'r':
            score = recall
        else:
            score = f_score
    return score







def _lcs_table(ref, can):
  """Create 2-d LCS score table."""
  rows = len(ref)
  cols = len(can)
  lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
  for i in range(1, rows + 1):
    for j in range(1, cols + 1):
      if ref[i - 1] == can[j - 1]:
        lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
      else:
        lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
  return lcs_table


def _backtrack_norec(t, ref, can):
  """Read out LCS."""
  i = len(ref)
  j = len(can)
  lcs = []
  while i > 0 and j > 0:
    if ref[i - 1] == can[j - 1]:
      lcs.insert(0, i-1)
      i -= 1
      j -= 1
    elif t[i][j - 1] > t[i - 1][j]:
      j -= 1
    else:
      i -= 1
  return lcs


def _summary_level_lcs(ref_sent, can_sent,weights):
  """ROUGE: Summary-level LCS, section 3.2 in ROUGE paper.
  Args:
    ref_sent: list of tokenized reference sentences
    can_sent: list of tokenized candidate sentences
  Returns:
    summary level ROUGE score
  """
  if not ref_sent or not can_sent:
    return 0

  m = sum(map(len, ref_sent))

  n = sum(map(len, can_sent))
  if not n or not m:
    return 0

  # get token counts to prevent double counting
  token_cnts_r = collections.Counter()
  token_cnts_c = collections.Counter()
  for s in ref_sent:
    # s is a list of tokens
    token_cnts_r.update(s)
  for s in can_sent:
    token_cnts_c.update(s)

  hits = 0
  i=0
  for r in ref_sent:
    lcs = _union_lcs(r, can_sent)
    # print(r)
    # print('LCS: {}'.format(lcs))

    # hits=hits+len(_union_lcs(r,can_sent))
    # Prevent double-counting:
    # The paper describes just computing hits += len(_union_lcs()),
    # but the implementation prevents double counting. We also
    # implement this as in version 1.5.5.
    for t in lcs:
      if token_cnts_c[t] > 0 and token_cnts_r[t] > 0:
        hits = hits+(1*weights[i])
        # hits=hits+1
        # print(weights[i])
        # print('weight for {} is {}'.format(i,weights[i]))

        token_cnts_c[t] -= 1
        token_cnts_r[t] -= 1
    i=i+1

  recall = hits / m
  precision = hits / n
  # fmeasure = scoring.fmeasure(precision, recall)
  return recall,precision


def _union_lcs(ref, c_list):
  """Find union LCS between a ref sentence and list of candidate sentences.
  Args:
    ref: list of tokens
    c_list: list of list of indices for LCS into reference summary
  Returns:
    List of tokens in ref representing union LCS.
  """
  lcs_list = [lcs_ind(ref, c) for c in c_list]
  return [ref[i] for i in _find_union(lcs_list)]


def _find_union(lcs_list):
  """Finds union LCS given a list of LCS."""
  return sorted(list(set().union(*lcs_list)))


def lcs_ind(ref, can):
  """Returns one of the longest lcs."""
  t = _lcs_table(ref, can)        # N        # Note: Does not support multi-line text.ote: Does not support multi-line text.

  return _backtrack_norec(t, ref, can)



def sentencelevelrouge(gold_summ,predicted_summ,N,weights):

    n=len(tokenizef(predicted_summ,True))
    m=len(tokenizef(gold_summ,True))
    # predicted_summ_sent=tokenize.sent_tokenize(predicted_summ)
    # gold_summ_sent=tokenize.sent_tokenize(gold_summ)
    # print(predicted_summ_sent)

    predicted_summ_sent = six.ensure_str(predicted_summ).split(". ")
    gold_summ_sent= six.ensure_str(gold_summ).split(". ")

    predicted_summ_sent2=[]
    for s in predicted_summ_sent:
      if len(s)!=0:
        predicted_summ_sent2.append(s)


    gold_summ_sent2=[]
    for s in gold_summ_sent:
      if len(s)!=0:
        gold_summ_sent2.append(s)


    finalmatch=0


    summ_grams_list=[]
    m2=0
    n2=0

    for sent in predicted_summ_sent2:
      tokens1=tokenizef(sent,True)
      summ_grams=Counter(make_n_grams(tokens1,N))
      n2=n2+len(summ_grams)
      summ_grams_list.append(summ_grams)

    ref_grams_list=[]


    for sent in gold_summ_sent2:
      tokens2=tokenizef(sent,True)
      ref_grams=Counter(make_n_grams(tokens2,N))
      m2=m2+len(ref_grams)
      ref_grams_list.append(ref_grams)

    # print(ref_grams_list)
    finalmatch2=0
    for summ_grams in summ_grams_list:
      countmatch=0
      countmatch2=0
      i=0
      for ref_grams in ref_grams_list:
        grams = min(summ_grams, ref_grams, key=len)
        # print(grams)
        count = sum(min(summ_grams[g], ref_grams[g]) for g in grams)

        for g in grams:
          x=min(summ_grams[g],ref_grams[g])
          summ_grams[g]=summ_grams[g]-x
          ref_grams[g]=ref_grams[g]-x

        countmatch=countmatch+(count*weights[i])
        # print('------------------')
        # print('weighted {}'.format(countmatch))
        # print('-----------------------')
        # countmatch2=countmatch2+(count)
        # print(countmatch2)

        # countmatch=countmatch+count
        # print(weights[i])
        i=i+1
      finalmatch2=finalmatch2+countmatch

    # print(finalmatch2)
    # print(finalmatch2/m)
    recall=finalmatch2/m2
    precision=finalmatch2/n2
    # print('m {}'.format(m))
    # print('m2{}'.format(m2))
    if recall==0 and precision==0:
        fmeasure=0
    else:
        fmeasure=2*((precision*recall)/(precision+recall))

    return recall,precision,fmeasure



def sentencelevelrougeL(gold_summ,predicted_summ,weights):
    predicted_summ_sent = six.ensure_str(predicted_summ).split(". ")
    predicted_summ_sent2=[]
    for s in predicted_summ_sent:
      if len(s)!=0:
        predicted_summ_sent2.append(tokenizef(s,True))


    gold_summ_sent= six.ensure_str(gold_summ).split(". ")
    gold_summ_sent2=[]
    for s in gold_summ_sent:
      if len(s)!=0:

        gold_summ_sent2.append(tokenizef(s,True))


    # print('pre: {}'.format(len(predicted_summ_sent2)))

    recall,precision = _summary_level_lcs(gold_summ_sent2,predicted_summ_sent2,weights)
    if precision==0 and recall==0:
        fmeasure=0
    else:
        fmeasure=2*((precision*recall)/(precision+recall))
    return recall,precision,fmeasure



def calculateWcov(references,article):
  Wcov=[]

  for lines1 in references:

    currwcov=0
    for lines2 in article:

      # rouge = Rouge()
      # scores = rouge.get_scores(lines1, lines2)
      # similarity=scores[0]['rouge-1']['r']
      scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
      scores2 = scorer.score(lines2,
                            lines1)
      similarity=scores2['rougeL'][1]
      # print(similarity)

      if similarity>=0.10:  #if similarity>0.10:
        # print(similarity)
        currwcov=currwcov+1
      else:
        # print(similarity)
        currwcov=currwcov+0
    
    # print(currwcov)
    # print(currwcov)
    currwcov=currwcov/(len(article))
    # print(currwcov)
    Wcov.append(currwcov)


  # print(Wcov)
  WcovSoftmax=softmax(Wcov)
  # print(WcovSoftmax)

  # print(WcovSoftmax.sum())

  # for w in WcovSoftmax:
  #     print(w)

  return WcovSoftmax,Wcov

# def calculateWcov(references,article):
#   Wcov=[]

#   for lines1 in references:

#     currwcov=0
#     for lines2 in article:

#       # rouge = Rouge()
#       # scores = rouge.get_scores(lines1, lines2)
#       # similarity=scores[0]['rouge-1']['r']
#       scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#       scores2 = scorer.score(lines2,
#                             lines1)
#       similarity=scores2['rougeL'][1]
#       # print(similarity)
#       currwcov=currwcov+similarity

#       # if similarity>0.35:  #if similarity>0.05:
#       #   # print(similarity)
#       #   currwcov=currwcov+1
#       # else:
#       #   # print(similarity)
#       #   currwcov=currwcov+0
#     # print(currwcov)
#     # print(currwcov)
#     currwcov=currwcov/(len(article))
#     # print(currwcov)
#     Wcov.append(currwcov)


#   # print(Wcov)
#   WcovSoftmax=softmax(Wcov)
#   # print(WcovSoftmax)

#   # print(WcovSoftmax.sum())

#   # for w in WcovSoftmax:
#   #     print(w)

#   return WcovSoftmax,Wcov

# def calculateWred(references):
#   Wred=[]

#   if len(references)==1:
#     # print('------')
#     Wred=[1]
#     WredSoftmax=softmax(Wred)
#     return WredSoftmax,Wred



#   for curridx in range(len(references)):
#     currwred=0
#     for otheridx in range(len(references)):
#       if curridx!=otheridx:
#         currsen=references[curridx]
#         othersen=references[otheridx]
#         # rouge = Rouge()
#         # scores = rouge.get_scores(othersen, currsen)
#         # similarity=scores[0]['rouge-1']['r']
#         scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#         scores2 = scorer.score(currsen,
#                               othersen)
#         similarity=scores2['rougeL'][1]
#         currwred=currwred+similarity

#     currwred=currwred/(len(references)-1)
#     currwred=1-currwred
#     Wred.append(currwred)



#   # print(Wred)
#   WredSoftmax=softmax(Wred)
#   # print(WredSoftmax.sum())
#   # for w in WredSoftmax:
#   #     print(w)

#   return WredSoftmax,Wred



def calculateWred(references):
  Wred=[]

  if len(references)==1:
    # print('------')
    Wred=[1]
    WredSoftmax=softmax(Wred)
    return WredSoftmax,Wred



  for curridx in range(len(references)):
    currwred=0
    for otheridx in range(len(references)):
      if curridx!=otheridx:
        currsen=references[curridx]
        othersen=references[otheridx]
        # rouge = Rouge()
        # scores = rouge.get_scores(othersen, currsen)
        # similarity=scores[0]['rouge-1']['r']
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores2 = scorer.score(currsen,
                              othersen)
        similarity=scores2['rougeL'][1]
        if similarity>=0.30:  #similarity>0.20
          currwred=currwred+1
        else:
          currwred=currwred+0
        # currwred=currwred+similarity

    currwred=currwred/(len(references)-1)
    currwred=1-currwred
    Wred.append(currwred)



  # print(Wred)
  WredSoftmax=softmax(Wred)
  # print(WredSoftmax.sum())
  # for w in WredSoftmax:
  #     print(w)

  return WredSoftmax,Wred


   # [0.33241695 0.36405289 0.30353016]






def weightedrouge(ref,decoded,article,N):
  predicted_summ_sent = six.ensure_str(decoded).split(". ")
  gold_summ_sent= six.ensure_str(ref).split(". ")
  article_Sent=six.ensure_str(article).split(". ")
  # print(len(article_Sent))
  # for l in article_Sent:
  #   print(l)
  #   print('-------------------')
  # print(predicted_summ_sent)
  # print(len(predicted_summ_sent))
  # print(gold_summ_sent)
  gold_summ_sent2=[]
  for line in gold_summ_sent:
    if len(line)!=0:
      gold_summ_sent2.append(line)


  predicted_summ_sent2=[]
  for line in predicted_summ_sent2:
    if len(line)!=0:
      predicted_summ_sent2.append(line)


  article_Sent2=[]
  for line in article_Sent:
    if len(line)!=0:
      if line.isspace() is False:
        article_Sent2.append(line.strip())


  # print(len(article_Sent2))
  # for line in article_Sent2:
  #   print(line)
  #   print('----------------')

  # print(gold_summ_sent2)

  Wcov,Wcovnotsm=calculateWcov(gold_summ_sent2,article_Sent2)

  # print(Wcov)
  # print(Wcov)
  Wred,Wrednotsm=calculateWred(gold_summ_sent2)
  # print(Wrednotsm)
  # print(Wred)

  # W = [(Wrednotsm[i] + Wcovnotsm[i])/2 for i in range(len(Wred))]
  # print(W)
  # print('------------------')
  W2 = [(((Wrednotsm[i] + Wcovnotsm[i])/2)*len(gold_summ_sent2)) for i in range(len(Wred))]
  # W2 = [(((Wred[i] + Wcov[i])/2)*len(gold_summ_sent2)) for i in range(len(Wred))]

  # W2 = [(((Wrednotsm[i] + 0)/1)*len(gold_summ_sent2)) for i in range(len(Wred))]
  # print(': {}'.format(Wcovnotsm))
  # print('W2: {}'.format(W2))

  # print(W2)
  # W2 = [(((Wrednotsm[i]+Wcovnotsm[i])/2)) for i in range(len(Wred))]
  # W2 = [(((2)/2)) for i in range(len(Wred))]
  # print('W3: {}'.format(W3))

  # print(W2)

  # W2 = [(((Wred[i] + 0)/1)*len(gold_summ_sent2)) for i in range(len(Wred))]


  # print(W)
  # file1 = open("ref2.txt","r")
  # ref=file1.read()

  # file2 = open('decoded2.txt','r')
  # decoded=file2.read()

  # file3 = open('article1.txt','r')
  # article=file3.read()
  # print(sentencelevelrouge(ref,decoded,1,W))

  # rouge = Rouge()
  # scores = rouge.get_scores(decoded, ref)
  # similarity=scores[0]['rouge-1']['r']
  # print(similarity)
  # originalrouge.append(compute_rouge_n(tokenizef(decoded,True),tokenizef(ref,True),1,'r'))




  # scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
  # scorer2 = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

  # scores2 = scorer.score(ref,
  #                       decoded)

  # print(scores2['rouge1'][1])


  # print(sentencelevelrougeL(ref,decoded,W))



  # scores3=scorer.score(article,decoded)
  # # print('rouge1 b/w article and decoded: {}'.format(scores3['rouge1'][1]))
  # scores4=scorer2.score(article,ref)
  # # print('rouge1 b/w artcile and ref: {}'.format(scores4['rouge1'][1]))
  # lamda=scores4['rougeL'][1]
  # r,p,f=sentencelevelrouge(ref,decoded,1,W)
  if(N!='l'):
    r2,p2,f2=sentencelevelrouge(ref,decoded,N,W2)
  else:
    r2,p2,f2=sentencelevelrougeL(ref,decoded,W2)


  
  # r2,p2,f2=sentencelevelrougeL(ref,decoded,W2)
  
  # print(((lamda)*r))
  # finaleval=((r)+((1-lamda)*scores3['rouge1'][1]))
  return r2,p2,f2,Wcovnotsm
  # print(finaleval)
  # finalevalmetric.append(finaleval)


  # finalevalmetricwithoutC.append(r2)






# finalevalmetric=[]
# originalrouge=[]
# finalevalmetricwithoutC=[]

# for i in range(0,1600):


#   curridx=i
#   idealidx=''
#   idealidx=idealidx+str(curridx)
#   rempos=6-len(idealidx)
#   for i in range(rempos):
#     idealidx='0'+idealidx

#   print(idealidx)
#   ref_add='/Data/anubhavcs17/asurl/summeval/SummEval/references/'+idealidx+'_reference.txt'
#   decoded_add='/Data/anubhavcs17/asurl/summeval/SummEval/decoded/'+idealidx+'_decoded.txt'
#   article_add='/Data/anubhavcs17/asurl/summeval/SummEval/articles/'+idealidx+'_article.txt'
#   file1 = open(ref_add,"r")
#   ref=file1.read()

#   file2 = open(decoded_add,'r')
#   decoded=file2.read()

#   file3 = open(article_add,'r')
#   article=file3.read()


#   predicted_summ_sent = six.ensure_str(decoded).split("\n")
#   gold_summ_sent= six.ensure_str(ref).split("\n")
#   article_Sent=six.ensure_str(article).split(" .")
#   # print(len(article_Sent))
#   # for l in article_Sent:
#   #   print(l)
#   #   print('-------------------')
#   # print(predicted_summ_sent)
#   # print(len(predicted_summ_sent))
#   # print(gold_summ_sent)
#   gold_summ_sent2=[]
#   for line in gold_summ_sent:
#     if len(line)!=0:
#       gold_summ_sent2.append(line)


#   predicted_summ_sent2=[]
#   for line in predicted_summ_sent2:
#     if len(line)!=0:
#       predicted_summ_sent2.append(line)


#   article_Sent2=[]
#   for line in article_Sent:
#     if len(line)!=0:
#       if line.isspace() is False:
#         article_Sent2.append(line.strip())


#   # print(len(article_Sent2))
#   # for line in article_Sent2:
#   #   print(line)
#   #   print('----------------')

#   Wcov,Wcovnotsm=calculateWcov(gold_summ_sent2,article_Sent2)
#   # print(Wcov)
#   # print(Wcov)
#   Wred=calculateWred(gold_summ_sent2)
#   # print(Wred)

#   W = [(Wred[i] + Wcov[i])/2 for i in range(len(Wred))]
#   W2 = [(((Wred[i] + Wcov[i])/2)*len(gold_summ_sent2)) for i in range(len(Wred))]

#   # print(W)
#   # file1 = open("ref2.txt","r")
#   # ref=file1.read()

#   # file2 = open('decoded2.txt','r')
#   # decoded=file2.read()

#   # file3 = open('article1.txt','r')
#   # article=file3.read()
#   # print(sentencelevelrouge(ref,decoded,1,W))

#   # rouge = Rouge()
#   # scores = rouge.get_scores(decoded, ref)
#   # similarity=scores[0]['rouge-1']['r']
#   # print(similarity)
#   originalrouge.append(compute_rouge_n(tokenizef(decoded,True),tokenizef(ref,True),1,'r'))




#   scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
#   scores2 = scorer.score(ref,
#                         decoded)

#   # print(scores2['rouge1'][1])


#   # print(sentencelevelrougeL(ref,decoded,W))



#   scores3=scorer.score(article,decoded)
#   # print('rouge1 b/w article and decoded: {}'.format(scores3['rouge1'][1]))
#   scores4=scorer.score(article,ref)
#   # print('rouge1 b/w artcile and ref: {}'.format(scores4['rouge1'][1]))
#   lamda=scores4['rouge1'][1]
#   r,p,f=sentencelevelrouge(ref,decoded,1,W)
#   # print(((lamda)*r))
#   finaleval=((r)+((1-lamda)*scores3['rouge1'][1]))
#   # print(finaleval)
#   finalevalmetric.append(finaleval)

#   r2,p2,f2=sentencelevelrouge(ref,decoded,1,W2)
#   finalevalmetricwithoutC.append(r2)


# # print(len(finalevalmetric))

# avgrouge=statistics.mean(originalrouge)
# avgeval=statistics.mean(finalevalmetric)
# avgfinalevalwithoutC=statistics.mean(finalevalmetricwithoutC)
# print('avg rouge: {}'.format(avgrouge))
# print('avg eval: {}'.format(avgeval))
# print('avg eval without C: {}'.format(avgfinalevalwithoutC))


# corr1, _ = pearsonr(originalrouge, finalevalmetricwithoutC)
# print('Pearsons correlation: {:.3f}'.format(corr1))
# corr2, _ = spearmanr(originalrouge, finalevalmetricwithoutC)
# print('Spearmans correlation: {:.3f}'.format(corr2))
# corr3, _ = kendalltau(originalrouge, finalevalmetricwithoutC)
# print('Kendall Rank correlation: {:.3f}'.format(corr3))


# with open("/Data/anubhavcs17/asurl/summeval/SummEval/fluency.txt", "rb") as fp:
#   fluency = pickle.load(fp)


# corr1, _ = pearsonr(fluency, finalevalmetricwithoutC)
# print('Pearsons correlation with consistency: {:.3f}'.format(corr1))
# corr2, _ = spearmanr(fluency, finalevalmetricwithoutC)
# print('Spearmans correlation with consistency: {:.3f}'.format(corr2))
# corr3, _ = kendalltau(fluency, finalevalmetricwithoutC)
# print('Kendall Rank correlation with consistency: {:.3f}'.format(corr3))

# corr1, _ = pearsonr(fluency, originalrouge)
# print('Pearsons correlation with consistency: {:.3f}'.format(corr1))
# corr2, _ = spearmanr(fluency, originalrouge)
# print('Spearmans correlation with consistency: {:.3f}'.format(corr2))
# corr3, _ = kendalltau(fluency, originalrouge)
# print('Kendall Rank correlation with consistency: {:.3f}'.format(corr3))

# corr1, _ = pearsonr(fluency, finalevalmetric)
# print('Pearsons correlation with consistency: {:.3f}'.format(corr1))
# corr2, _ = spearmanr(fluency, finalevalmetric)
# print('Spearmans correlation with consistency: {:.3f}'.format(corr2))
# corr3, _ = kendalltau(fluency, finalevalmetric)
# print('Kendall Rank correlation with consistency: {:.3f}'.format(corr3))

# with open("test005.txt", "rb") as fp:
#   finalevalmetric = pickle.load(fp)

# print(statistics.mean(finalevalmetric))


# with open('testoriginal.txt','rb') as fp:
#   originalrouge=pickle.load(fp)

# print(statistics.mean(originalrouge))


# corr1, _ = pearsonr(originalrouge, finalevalmetric)
# print('Pearsons correlation: {:.3f}'.format(corr1))
# corr2, _ = spearmanr(originalrouge, finalevalmetric)
# print('Spearmans correlation: {:.3f}'.format(corr2))
# corr3, _ = kendalltau(originalrouge, finalevalmetric)
# print('Kendall Rank correlation: {:.3f}'.format(corr3))

# with open("testoriginal4.txt", "wb") as fp:
#   pickle.dump(originalrouge, fp)


# with open("test.txt", "wb") as fp:
#   pickle.dump(finalevalmetric, fp)
# Kendall Rank correlation b/w coherence and our final rouge-l fscore: 0.147
# Kendall Rank correlation b/w coherence and OG rouge-l fscore: 0.109
# Kendall Rank correlation b/w consistency and our final rouge-l fscore: 0.180
# kendall Rank correlation b/w consistency and OG rouge-l fscore: 0.090
# kenda rank correlation b/w fluency and our final rouge-l fscore: 0.128
# kenda rank correlation b/w fluency and OG rouge-l fscore: 0.067
# kenda rank correlation b/w relevance and our final rouge-l fscore: 0.247
# kenda rank correlation b/w relevance and OG rouge-l fscore: 0.216