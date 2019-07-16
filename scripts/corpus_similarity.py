'''
Corpus similarity metrics
'''
import pandas as pd
import numpy as np
import re
import math

import scipy
from scipy.special import rel_entr

import collections
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer 

from gensim.models import word2vec

class Corpus():

	def __init__ (self, df):
		self.df = df

	def sample(self, name, sep=','):
		'''
		create 1000 instances sample from the dataset
		'''
		if sep == '\t':
			return self.df.sample(1000).to_csv('1000_'+name+'.tsv', sep = sep)

		else:
			return self.df.sample(1000).to_csv('1000_'+name+'.csv', sep = sep)

	def preprocessing(self, column):
		'''
		text preprocessing and tokenization
		'''
		doc = self.df[column].tolist()

		self.doc_token = []

		stop_words = set(stopwords.words('english'))

		for text in doc:

			text = text.lower()

			text_nolink = re.sub(r"https://.*?/[\dA-Za-z]+", "", text)
			text_notag = re.sub(r"#[\dA-Za-z]+", "", text_nolink)
			text_nouser = re.sub(r"@[\dA-Za-z_]+ ", "", text_notag)
			text_letters = re.sub("[^a-zA-Z]", " ", text_nouser) 
		#text_nonumber = re.sub(r"\d+", "", text_nouser)
		#text_nopunct = re.sub(r"[!”#$%&’()–*+,-./:;<=>?@[\]'^_`{|}~]❤️", " ", text_nonumber)

			text_tokenize = word_tokenize(text_letters)
			text_clean = [word for word in text_tokenize if word not in stop_words]
			self.doc_token.append(text_clean)
		
		return self.doc_token

	def join(self):
		'''
		join list of tokens
		'''
		text = [word for sentence in self.doc_token for word in sentence]
		self.full_text = ' '.join(text)
		return self.full_text

	def cosine_similarity (self, model, doc2, doc3):
		'''
		cosine similarity between documents
		'''
		vector_1 = np.mean([model[word] for word in self.full_text.split() if word in model],axis=0)
		vector_2 = np.mean([model[word] for word in doc2.split() if word in model],axis=0)
		vector_3 = np.mean([model[word] for word in doc3.split() if word in model],axis=0)
		cosine = cosine_similarity([vector_1, vector_2, vector_3], [vector_1, vector_2, vector_3])

		df_cosine = pd.DataFrame(cosine,columns=['Wassem & Hovy', 'Offense Eval', 'Hate Eval'],
			index=['Wassem & Hovy', 'Offense Eval', 'Hate Eval'])

		print(df_cosine)

	def token_count(self):
		'''
		create dictionary with tokens frequency
		'''
		self.tokens = collections.defaultdict(lambda: 0.)
		for m in re.finditer(r"(\w+)", self.full_text, re.UNICODE):
			m = m.group(1).lower()
			if len(m) < 2:
				continue
			if m in stopwords.words('english'):
				continue
			self.tokens[m] += 1
		return self.tokens 

	def kldiv(self, _s, _t):
		'''
		Kullback–Leibler divergence
		'''
		if (len(_s) == 0):
			return 1e33

		if (len(_t) == 0):
			return 1e33

		ssum = 0. + sum(_s.values())
		slen = len(_s)

		tsum = 0. + sum(_t.values())
		tlen = len(_t)

		vocabdiff = set(_s.keys()).difference(set(_t.keys()))
		lenvocabdiff = len(vocabdiff)

		''' epsilon ''' 
		epsilon = min(min(_s.values())/ssum, min(_t.values())/tsum) * 0.001

		''' gamma ''' 
		gamma = 1 - lenvocabdiff * epsilon

		''' Check if distribution probabilities sum to 1'''
		sc = sum([v/ssum for v in list(_s.values())])
		st = sum([v/tsum for v in list(_t.values())])

		vocab = Counter(_s) + Counter(_t)
		ps = []
		pt = []
		for t, v in list(vocab.items()):
			if t in _s:
				pts = gamma * (_s[t] / ssum)
			else:
				pts = epsilon

			if t in _t:
				ptt = gamma * (_t[t] / tsum)
			else:
				ptt = epsilon

			ps.append(pts)
			pt.append(ptt)

		return ps, pt


	def jensen_shannon_divergence(self, doc2):
		'''Calculates Jensen-Shannon divergence (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence).'''
		'''from https://github.com/sebastianruder/learn-to-select-data/blob/master/similarity.py '''
		kl = self.kldiv(self.tokens , doc2)
		d1_ = kl[0]
		d2_ = kl[1]

		repr1 = np.asarray(d1_)
		repr2 = np.asarray(d2_)

		avg_repr = 0.5 * (repr1 + repr2)
		sim = 1 - 0.5 * (scipy.stats.entropy(repr1, avg_repr) + scipy.stats.entropy(repr2, avg_repr))
		if np.isinf(sim):
		# the similarity is -inf if no term in the document is in the vocabulary
			return 0
		return sim

	def out_domain(self, corpus_2):
		'''
		OOV rate 
		'''
		corpus_1 = set(self.full_text.split())
		corpus_2 = set(corpus_2.split())
		count = 0
		for word in corpus_1:
			if word in corpus_2:
				count += 1
			else:
				pass
		cor_1_percent = (count/len(corpus_1))*100
		cor_2_percent = (count/len(corpus_2))*100
		return 100-((len(corpus_1) - count)/len(corpus_1))*100




def loadGloveModel(gloveFile):
	'''
	loading Glove model
	'''
	print("Loading Glove Model")

	f = open(gloveFile,'r')

	model = {}

	for line in f:
		splitLine = line.split()
		word = splitLine[0]
		embedding = np.array([float(val) for val in splitLine[1:]])
		model[word] = embedding

	f.close()

	print("Done.",len(model)," words loaded!")

	return model


if __name__ == '__main__':
	# (Waseem & Hovy, 2016)
	wh_df = pd.read_csv('./datasets/2016_waseem_hovy_original.csv', sep = ',', engine='python', header = None)
	wh_df = wh_df.dropna()
	wh_df = wh_df.loc[wh_df[2] != 'none']

	# HatEval 2019
	hate_df = pd.read_csv('./datasets/hateval-test/train_en.tsv', sep = '\t')
	hate_df = hate_df.loc[hate_df['HS'] == 1]

	# OffensEval 2019
	offense_df = pd.read_csv('./datasets/offense_eval/off_eval.tsv', sep = '\t')
	offense_df = offense_df.loc[offense_df['subtask_a'] != 'NOT']

	wh = Corpus(wh_df)
	hate = Corpus(hate_df)
	offense = Corpus(offense_df)

	# tokenization
	wh.preprocessing(1)
	wh_full = wh.join()

	hate.preprocessing('text')
	hate_full = hate.join()

	offense.preprocessing('tweet')
	offense_full = offense.join()



	# cosine similarity
	model = loadGloveModel('./twitter_model/glove.twitter.27B.50d.txt')

	print('Cosine similarity')
	wh.cosine_similarity(model, offense_full, hate_full)

	print('\nJensen-Shannon divergence')

	wh.token_count()

	print('Waseem & Hovy with OffensEval: ', wh.jensen_shannon_divergence(offense.token_count()))
	print('Waseem & Hovy with HatEval: ', wh.jensen_shannon_divergence(hate.token_count()))
	print('OffensEval with HatEval: ', offense.jensen_shannon_divergence(hate.token_count()))

	print('\nOut of domain vocabularly')

	print('Waseem & Hovy with OffensEval: ', wh.out_domain(offense_full))
	print('OffensEval with Waseem & Hovy: ', offense.out_domain(wh_full))

	print('Waseem & Hovy with HatEval: ', wh.out_domain(hate_full))
	print('HatEval with Waseem & Hovy: ', hate.out_domain(wh_full))

	print('OffensEval with HatEval: ', offense.out_domain(hate_full))
	print('HatEval with OffensEval: ', hate.out_domain(offense_full))




