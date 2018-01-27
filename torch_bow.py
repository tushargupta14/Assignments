## bag of Words classifier

import torch as th 
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F

import torch.optim as optim 

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

# word_to_ix maps each word in the vocab to a unique integer, which will be its
# index into the Bag of words vector
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2

class BagofWordsClassifier(nn.Module) :

	def __init__(self, NUM_LABELS,VOCAB_SIZE) :

		super(BagofWordsClassifier,self).__init__()

		self.Linear = nn.Linear(VOCAB_SIZE,NUM_LABELS)


	def forward(self,bow_vec) :

		return F.softmax(self.linear(bow_vec))






def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


model = BagofWordsClassifier(NUM_LABELS,VOCAB_SIZE)

for param in model.parameters() :

	print param 





