# Create vectors using the bag of words model and assigns labels to the functions
# Improvemnets:

# Selecting a smaller feature word list for dense representation 
# SVD on current vectors 
# RNN model
# Treating this as a muti-label classification problem. Currently a multi-class classification problem.

import json 
import csv 
import numpy as np 
import cPickle as cp 
from collections import defaultdict
def build_vocabulary() :

	word_idf_vocab = {}

	feature_word_list = []

	num_docs = 0
	with open("data/dataset.csv","rb+") as f :

		reader = csv.reader(f)

		for row in reader :
			description = row[2]
			num_docs +=1 
			for word in description.split() :
				if word not in word_idf_vocab :
					word_idf_vocab[word] =1 
				elif word in word_idf_vocab :
					word_idf_vocab[word]+=1 

	for k in word_idf_vocab.iterkeys() :

		word_idf_vocab[k] = np.log( 1 + ( float(num_docs)/word_idf_vocab[k]))
		feature_word_list.append(k)	
	
	print num_docs,"Num Docs"
	print len(word_idf_vocab)
	#with open("word_idf_values.json","wb+") as f :
		#json.dump(word_idf_vocab,f)

	print len(feature_word_list),"List Length"

	with open("data/feature_word_list.pkl","wb+") as out_file:

		cp.dump(feature_word_list,out_file)

feature_word_list = cp.load(open("data/feature_word_list.pkl"))
word_idf_vocab = json.load(open("data/word_idf_values.json"))
		

def create_vec(description,word_idf_vocab) :

	vector = [0 for x in range(len(word_idf_vocab))]

	term_frequency_dict = {}
	for word in description.split() :
		if word in term_frequency_dict : 

			term_frequency_dict[word]+=1 
		elif word not in term_frequency_dict :
			term_frequency_dict[word] =1 

	
	for word in term_frequency_dict :
		#print term_frequency_dict[word]

		idx = feature_word_list.index(word)

		vector[idx] = word_idf_vocab[word] * term_frequency_dict[word]


	return vector
def tfidf_for_docs() :

	with open("data/word_idf_values.json","rb+") as f :
		word_idf_vocab = json.load(f)

	tfidf_dict = defaultdict(list)



	count =0 
	with open("data/dataset.csv","rb+") as f :

		reader = csv.reader(f)

		for row in reader : 
			description = row[2]

			des_vec = create_vec(description,word_idf_vocab)

			tfidf_dict[count] = des_vec
			print count 

			count+=1 	

	print "Dumping to dictionary"
	with open("doc_vectors.json","wb+") as fp :
		json.dump(tfidf_dict,fp)

def assign_labels():
	
	function_label_mapping = json.load(open("data/function_labels.json"))
	doc_label_dict = defaultdict(list)
	with open("data/dataset.csv","rb+") as f :

		reader = csv.reader(f)
		count =0 
		for row in reader : 

			functions = row[0]
			for func in functions.split(",") :
				#print func 
				label = function_label_mapping[func]

				doc_label_dict[count].append(label)
			print count 
			count+=1 

	with open("data/doc_labels.json","wb+") as fp :

		json.dump(doc_label_dict,fp)
if __name__ == "__main__" :

	build_vocabulary()
	create_feature_words()
	tfidf_for_docs()
	assign_labels()