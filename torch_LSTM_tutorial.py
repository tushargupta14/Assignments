
# coding: utf-8

# In[2]:

import torch 
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[15]:

x = [1,2,4]
V = torch.Tensor([1,2,3])


# In[17]:

data = [ ("me gusta comer en la cafeteria".split(), "SPANISH"),
         ("Give it to me".split(), "ENGLISH"),
         ("No creo que sea una buena idea".split(), "SPANISH"),
         ("No it is not a good idea to get lost at sea".split(), "ENGLISH") ]

test_data = [ ("Yo creo que si".split(), "SPANISH"),
              ("it is lost on me".split(), "ENGLISH")]


# In[18]:

word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print word_to_ix

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2


# In[22]:

vec = torch.zeros(3)
vec


# In[24]:

word_to_idx = {"hello" : 0 , "world" : 1}
embeds = nn.Embedding(len(word_to_idx),5)


# In[32]:

lookup_tensor = torch.LongTensor(word_to_idx["world"])


# In[34]:

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [ ([test_sentence[i], test_sentence[i+1]], test_sentence[i+2]) for i in xrange(len(test_sentence) - 2) ]
print trigrams[:3] # print the first 3, just so you can see what they look like


# In[35]:

vocab = set(test_sentence)
w_to_idx = {w:idx for idx,w in enumerate(vocab)}


# In[41]:

w_to_idx['When']


# In[36]:

class NGramLanguageModel(nn.Module) : 
    def __init__(self,vocab_size,embedding_dim,context_size):
        
        super(NGramLanguageModel,self).__init__()
        self.Embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.linear1 = nn.Linear(context_size*embedding_dim,128)
        self.linear2  = nn.Linear(128,vocab_size)
        
    def forward(self,inputs) :
        
        embeds = Embeddings(inputs).view((1,-1))
        lin1_output = F.relu(self.linear1(embeds))
        out = self.linear2(lin1_output)
        log_probs = F.log_softmax(out)
        return log_probs
        


# In[43]:

loss_function = nn.NLLLoss()
model = NGramLanguageModel(len(vocab),300,2)
optimizer = optim.SGD(model.parameters(),lr = 0.001)
for epoch in xrange(10):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
    
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = map(lambda w: word_to_idx[w], context)
        context_var = autograd.Variable( torch.LongTensor(context_idxs) )
    
        # Step 2. Recall that torch *accumulates* gradients.  Before passing in a new instance,
        # you need to zero out the gradients from the old instance
        model.zero_grad()
    
        # Step 3. Run the forward pass, getting log probabilities over next words
        log_probs = model(context_var)
    
        # Step 4. Compute your loss function. (Again, Torch wants the target word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))
    
        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()
    
        total_loss += loss.data
    losses.append(total_loss)
print losses # The loss decreased every iteration over the training data!


# In[46]:

hidden = (autograd.Variable(torch.randn(1,1,3)), autograd.Variable(torch.randn((1,1,3))))
hidden[0]


# In[48]:

def prepare_sequence(seq, to_ix):
    idxs = map(lambda w: to_ix[w], seq)
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print word_to_ix
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}


# In[50]:

embed_dim = 10 
hidden_dim = 10 


# In[ ]:

class LSTMtagger(nn.Module) :
    
    def __init__(self,embed_dim,hidden_dim,vocab_size,tagset_size):
        
        super(LSTMtagger,self).__init__() 
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
        
    def self.init_hidden(self) :
        
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
        


# In[52]:

torch.zeros(1,64,3)


# In[ ]:



