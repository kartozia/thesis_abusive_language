'''
OffensEval Experiments
'''
import argparse
import pandas as pd 
import numpy as np
import re
import statistics as stats
import gensim.models as gm
from gensim.models import Word2Vec, KeyedVectors
from scipy import sparse
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import features
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import shuffle
from sklearn.externals import joblib

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


from scipy.sparse import hstack, csr_matrix

SEED = 2048
stop_words = set(stopwords.words('english'))

def preprocessing(text):
	'''
	text preprocessing and tokenization
	'''
	text = text.lower()

	#text_nolink = re.sub(r"https://.*?/[\dA-Za-z]+", "", text)
	#text_notag = re.sub(r"#[\dA-Za-z]+", "", text_nolink)
	#text_nouser = re.sub(r"@[\dA-Za-z_]+ ", "", text_notag)
	text_letters = re.sub("[^a-zA-Z]", " ", text) 
		#text_nonumber = re.sub(r"\d+", "", text_nouser)
		#text_nopunct = re.sub(r"[!”#$%&’()–*+,-./:;<=>?@[\]'^_`{|}~]❤️", " ", text_nonumber)

	text_tokenize = word_tokenize(text_letters)
	text_clean = [word for word in text_tokenize if word not in stop_words]
	
	return text_clean

def load_embeddings(embedding_file):
	'''
	loading embeddings from file
	input: embeddings stored as json (json), pickle (pickle or p) or gensim model (bin)
	output: embeddings in a dict-like structure available for look-up, vocab covered by the embeddings as a set
	'''
	if embedding_file.endswith('json'):
		f = open(embedding_file, 'r', encoding='utf-8')
		embeds = json.load(f)
		f.close()
		vocab = {k for k,v in embeds.items()}
	elif embedding_file.endswith('bin'):
		embeds = gm.KeyedVectors.load(embedding_file).wv
		vocab = {word for word in embeds.index2word}
	elif embedding_file.endswith('p') or embedding_file.endswith('pickle'):
		f = open(embedding_file,'rb')
		embeds = pickle.load(f)
		f.close
		vocab = {k for k,v in embeds.items()}
	elif embedding_file.endswith('txt'):
		embeds = gm.KeyedVectors.load_word2vec_format(embedding_file, binary=False)
		#vocab = embeds.wv.vocab

	return embeds

def offensive(tweet, list_):
	count = 0
	for token in tweet:
		
		if token in list_:
			count += 1
		else:
			pass
	if count > 0:
		return count/len(tweet)*100
	else:
		return 0

def offensive_advanced(tweet, list_):
	#count = 0
	ph = []
	for phrase in list_:
		for token in tweet:
			if (len(phrase) > 1) and (len(tweet)>1):
				if token in phrase:
					if (tweet[tweet.index(token)] == tweet[-1]) and (tweet[tweet.index(token)-1] in phrase):
						ph.append(phrase)
					elif tweet[tweet.index(token)] != tweet[-1]:
						#print(token, tweet.index(token))
						if ((token in phrase) and (tweet[tweet.index(token)+1] in phrase))|\
						((token in phrase) and (tweet[tweet.index(token)-1] in phrase)):
							ph.append(phrase)
			elif (len(phrase) > 1) and (len(tweet) == 1):
				pass
			elif len(phrase) == 1:
				if "".join(phrase) == token:
					ph.append(phrase)
			else:
				pass	   
	ph_len = len(set(tuple(row) for row in ph))
	if ph_len >= 1:
		#print(count)
		return( ph_len/len(tweet)*100)
	else:
		return 0

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--approach', type=str, default='SVM')
	parser.add_argument('--labels', type=int, default=3)
	parser.add_argument('--embeddings', type=str, default='no')
	parser.add_argument('--word', type=int, default=None)
	parser.add_argument('--char', type=int, default=None)
	args = parser.parse_args()



	df_train = pd.read_csv('offense_train.tsv', sep = '\t')
	df_test = pd.read_csv('offense_test.tsv', sep = '\t')

	df_train['abuse'] = df_train['abuse'].apply(lambda x: str(x).lower())
	df_test['abuse'] = df_test['abuse'].apply(lambda x: str(x).lower())

	df_train = df_train.loc[df_train['subtask_a'] == 'OFF']
	df_test = df_test.loc[df_test['label'] == 'OFF']

	if args.labels == 2:
		df_train = df_train.loc[(df_train['abuse'] == 'explicit') | (df_train['abuse'] == 'implicit')]
		df_test = df_test.loc[(df_test['abuse'] == 'explicit') | (df_test['abuse'] == 'implicit')]
	else:
		pass

	x_train = df_train['tweet'].apply(lambda x: preprocessing(x))
	x_train = [' '.join(sublist) for sublist in x_train]
	y_train = df_train['abuse'].tolist()

	x_test = df_test['tweet'].apply(lambda x: preprocessing(x))
	x_test = [' '.join(sublist) for sublist in x_test]
	y_test = df_test['abuse'].tolist()

	if args.approach == 'SVM':
		if args.embeddings == 'no':
			tfidf_char = TfidfVectorizer(analyzer = 'char', ngram_range=(args.char, args.char))
			tfidf_char.fit(x_train)
			x_train_char = tfidf_char.transform(x_train)
			x_test_char = tfidf_char.transform(x_test)

			tfidf_word = TfidfVectorizer(analyzer = 'word', ngram_range=(args.word, args.word))
			tfidf_word.fit(x_train)
			x_train_word = tfidf_word.transform(x_train)
			x_test_word = tfidf_word.transform(x_test)

			x_train = sparse.hstack((x_train_word,x_train_char)).tocsr()
			x_test = sparse.hstack((x_test_word, x_test_char)).tocsr()

			clf = SVC(C=1.0, kernel='linear', gamma='auto', random_state = SEED)

			print('Fitting on training data...')
			clf.fit(x_train,y_train)
			print('Predicting...')
			predictions_SVM = clf.predict(x_test)
			print(f1_score(predictions_SVM, y_test, average = 'macro'))

			df_test['predicted'] = predictions_SVM
			error = df_test.loc[df_test['abuse'] != df_test['predicted']]
			error.to_csv('error_svm_multi.tsv', sep = '\t')
			print(df_test['abuse'].value_counts())
			print(df_test['predicted'].value_counts())

			print('F1-score for each label')
			print(precision_recall_fscore_support(predictions_SVM, y_test,labels=['explicit', 'implicit', 'not abusive']))

		elif args.embeddings == 'yes':
			tfidf_word = TfidfVectorizer(analyzer='word',
							 ngram_range=(args.word, args.word))
			tfidf_char = TfidfVectorizer(analyzer='char',
							 ngram_range=(args.char, args.char))

			path_to_embs = './glove_200_w2vformat.txt'
			print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
			embeddings = load_embeddings(path_to_embs)
			print('Done')


			vectorizer = FeatureUnion([('word', tfidf_word),
								('char', tfidf_char),
								('word_embeds', features.Embeddings(embeddings, pool='pool'))])

			clf = LinearSVC(random_state = SEED)

			classifier = Pipeline([
                     ('vectorize', vectorizer),
                     ('classify', clf)])

			print('Fitting on training data...')
			classifier.fit(x_train, y_train)

			print('Predicting...')

			predictions_SVM = classifier.predict(x_test)
			print(f1_score(predictions_SVM, y_test, average = 'macro'))


			df_test['predicted'] = predictions_SVM
			error = df_test.loc[df_test['abuse'] != df_test['predicted']]
			error.to_csv('error_svm_multi_embd.tsv', sep = '\t')
			print(df_test['abuse'].value_counts())
			print(df_test['predicted'].value_counts())

			print('F1-score for each label')
			print(precision_recall_fscore_support(predictions_SVM, y_test,labels=['explicit', 'implicit', 'not abusive']))



	elif args.approach == 'dictionary':
		hurt_en = pd.read_csv('./lists/hurtlex_EN_conservative.tsv', sep = '\t', header = None) # Hurtlex conservative
		hurt_list = hurt_en[2].tolist()
		hurt_list = [x.split() for x in hurt_list]
		hurt_list = [[word.lower() for word in sublist if word not in stop_words] for sublist in hurt_list]

		hurt_en_long = pd.read_csv('./lists/hurtlex_EN_inclusive.tsv', sep = '\t', header = None) # Hurlex inclusive
		hurt_list_long = hurt_en_long[2].tolist()
		hurt_list_long = [x.split() for x in hurt_list_long]
		hurt_list_long = [[word.lower() for word in sublist if word not in stop_words] for sublist in hurt_list_long]

		with open('./lists/baseLexicon.txt', 'r', encoding='utf-8') as f: # Wiegand base lexicon
			wiegand_list = f.readlines()
			wiegand_list = [word.split('_')[0] for word in wiegand_list]

		with open('./lists/expandedLexicon.txt', 'r', encoding='utf-8') as f: # Wiegand expanded lexicon
			wiegand_extended = f.readlines()
			wiegand_extended = [word.split('_')[0] for word in wiegand_extended]
		
		df_test['tweet_pre'] = df_test['tweet'].apply(lambda x: preprocessing(x))

		# Wiegand expanded
		df_test['predict'] = df_test['tweet_pre'].apply(lambda x: offensive(x, wiegand_extended))
		df_test['predict_label'] = df_test['predict'].apply(lambda x: 'explicit' if x > 10 else 'implicit')
		print('Wiegand expanded lexicon: ', \
			f1_score(df_test['predict_label'].tolist(), df_test['abuse'].tolist(), average = 'macro'))

		error = df_test.loc[df_test['abuse'] != df_test['predict_label']]
		error.to_csv('error_svm_dict.tsv', sep = '\t')
		print(df_test['abuse'].value_counts())
		print(df_test['predict_label'].value_counts())

		print('F1-score for each label')
		print(precision_recall_fscore_support(df_test['predict_label'].tolist(), df_test['abuse'].tolist(),labels=['explicit', 'implicit']))

		# Wiegand base
		df_test['predict'] = df_test['tweet_pre'].apply(lambda x: offensive(x, wiegand_list))
		df_test['predict_label'] = df_test['predict'].apply(lambda x: 'explicit' if x > 10 else 'implicit')
		print('Wiegand base lexicon: ', \
			f1_score(df_test['predict_label'].tolist(), df_test['abuse'].tolist(), average = 'macro'))

		# Hurtlex conservative
		df_test['predict'] = df_test['tweet_pre'].apply(lambda x: offensive_advanced(x, hurt_list))
		df_test['predict_label'] = df_test['predict'].apply(lambda x: 'explicit' if x > 10 else 'implicit')
		print('Hurlex conservative: ', \
			f1_score(df_test['predict_label'].tolist(), df_test['abuse'].tolist(), average = 'macro'))

		# Hurtlex inclusive
		df_test['predict'] = df_test['tweet_pre'].apply(lambda x: offensive_advanced(x, hurt_list_long))
		df_test['predict_label'] = df_test['predict'].apply(lambda x: 'explicit' if x > 10 else 'implicit')
		print('Hurlex inclusive: ', \
			f1_score(df_test['predict_label'].tolist(), df_test['abuse'].tolist(), average = 'macro'))

		# precision_recall_fscore_support(df_test['predict_label'].tolist(), df_test['abuse'].tolist(),labels=['epxlicit','implicit'])




