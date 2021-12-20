from rouge_score import rouge_scorer
import sys, glob, json

THRESHOLD = 0.1
article_path = '/Data/anubhavcs17/asurl/finished_files/test/'
N = 11490
N = 1149*2

def calculate_cov(references, article):
  scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
  cov = sum([sum([1 for art in article if scorer.score(summ, art)['rougeL'][1] >= THRESHOLD]) for summ in references])
  d = len(article)*len(references)
  return cov/d

def get_avg_cov(path, N=None, eval_gold=False):
	res = 0
	for file_id in range(N):
		with open('{}/{}.json'.format(article_path, file_id)) as f:
			data = json.load(f)
		art, abs = data['article'], data['abstract']
		if eval_gold:
			res += calculate_cov(abs, art)
		else:
			with open('{}/{}_decoded.txt'.format(path, str(file_id).zfill(6))) as f:
				refs = f.readlines()
			res += calculate_cov(refs, art)
	return res/N

for idx, path in enumerate(sys.argv[:]):
	print(get_avg_cov(path, N, idx==0))
