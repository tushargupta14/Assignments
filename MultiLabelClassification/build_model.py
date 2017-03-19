# Script trains a structured SVM classifier for the prediction task treating as a multi-label-classification problem
import json 
import csv
import numpy as np 
import cPickle as cp 
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse

from sklearn.metrics import hamming_loss
from sklearn.metrics import mutual_info_score
from scipy.sparse.csgraph import minimum_spanning_tree

from pystruct.learners import OneSlackSSVM
from pystruct.models import MultiLabelClf
from pystruct.datasets import load_scene
import itertools 

from sklearn import model_selection 
def load_data():

	# Creating the input to the model 
	X_Dict = json.load(open("data/doc_vectors.json"))
	Y_Dict  = json.load(open("data/doc_labels.json"))

	print "loaded"
	dims = len(X_Dict["0"])
	function_labels_dict = json.load(open("data/function_labels.json"))
	n_classes = len(function_labels_dict)

   	n_tot = 8349 
	x_data = np.zeros([n_tot, dims])
	count = 0 
	for k,v in X_Dict.iteritems():
		idx = int(k)
    	#print count 
    	x_data[idx,] = np.asarray(X_Dict[k])
    	#count+=1 
	
	count = 0 
	

	for k,v in Y_Dict.iteritems() :
		idx = int(k) 
    	count+=1 
    	#print v
    	y_data[idx,] = Y_Dict[k]
    	print count

	y_label_data = MultiLabelBinarizer().fit_transform(y_data)

	x_train = x_data[0:int(x_data.shape[0]*0.7)-1]

	x_test = x_data[int(x_data.shape[0]*0.7):]

	y_train = y_label_data[0:int(y_label_data.shape[0]*0.7)-1]

	y_test = y_label_data[int(y_label_data.shape[0]*0.7):]


	np.save("data/x_train.npy",x_train)
	np.save("data/x_test.npy",x_test)
	np.save("data/y_train.npy",y_train)
	np.save("data/y_test.npy",y_test)

	return x_train,y_train,x_test,y_test


def load_data1() :

	# Alternative function to load saved data 

	x_train = np.load("data/x_train.npy")
	x_test = np.load("data/x_test.npy")

	y_train= np.load("data/y_train.npy")
	y_test = np.load("data/y_test.npy")

	return x_train,y_train,x_test,y_test

def chow_liu_tree(y_):
    # compute mutual information using sklearn
    n_labels = y_.shape[1]
    mi = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            mi[i, j] = mutual_info_score(y_[:, i], y_[:, j])
    mst = minimum_spanning_tree(sparse.csr_matrix(-mi))
    edges = np.vstack(mst.nonzero()).T
    edges.sort(axis=1)
    return edges



def ssvm_classifier() :

	x_train,y_train,x_test,y_test = load_data1()

	print "Data Loaded"
	

	pca = PCA(n_components= 1000)
	x_train_reduced = pca.fit_transform(x_train)
	x_test_reduced = pca.fit_transform(x_test)

	print "PCA finished"

	print "Learning the model"

	n_labels = y_train.shape[1]

	full = np.vstack([x for x in itertools.combinations(range(n_labels), 2)])
	tree = chow_liu_tree(y_train)

	
	independent_model = MultiLabelClf(inference_method='unary')


	independent_ssvm = OneSlackSSVM(independent_model, C=.1, tol=0.01)
	independent_ssvm.fit(x_train_reduced, y_train)
	print "saving model ..."
	with open("data/independent_ssvm.pkl","wb+") as f :
		cp.dump(independent_ssvm,f)
	#print "Calculatin the cross-validation scores"
	#scores = model_selection.cross_val_score(independent_ssvm,x_train_reduced,y_train,cv=3)

	print independent_ssvm.score(x_test_reduced,y_test)

	

def OneVsRestClassifier() :
	x_train,y_train,x_test,y_test = load_data()
	#x_train,y_train,x_test,y_test = load_data1()

	print "Data Loaded"
	

	pca = PCA(n_components= 1000)
	x_train_reduced = pca.fit_transform(x_train)
	x_test_reduced = pca.fit_transform(x_test)

	print "PCA finished"

	print "Learning the model"

	clf = OneVsRestClassifier(SVC(kernel = "poly"))
	clf.fit(x_train_reduced,y_train)
	print "Calculating score..."
	clf.score(x_test_reduced,y_test)


if __name__ =="__main__" :

	ssvm_classifier()