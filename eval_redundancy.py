from rouge_score import rouge_scorer
import sys, glob, json

N = 4000

def compute_red(references):
  if len(references)==1:
    return 0, 0

  n = len(references)
  Wred, Wredt = 0, 0
  scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
  for i, refi in enumerate(references):
    for j, refj in enumerate(references):
      if i != j:
        similarity = scorer.score(refi, refj)['rougeL'][1]
        Wredt += similarity
        if similarity >= 0.30:  #similarity>0.20
          Wred += 1

  return Wred/n, Wredt/(n*(n-1))

def get_avg_red(path, N=None, gold=False):
	scorew, scoret = 0, 0
	for file_id in range(N):
		if gold:
			with open('/Data/anubhavcs17/asurl/finished_files/test/{}.json'.format(file_id)) as f:
				art = json.load(f)['abstract']
		else:
			with open('{}/{}_decoded.txt'.format(path, str(file_id).zfill(6))) as f:
				art = f.readlines()
		w, wt = compute_red(art)
		scorew += w
		scoret += wt
	return scorew/N, scoret/N

for idx, p in enumerate(sys.argv[:]):
	print(get_avg_red(p, N, idx==0))
