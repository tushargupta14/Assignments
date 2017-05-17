

import torch as th 
from torch.autograd import Variable

import torch.nn as nn 
#dtype = th.cuda.Tensor



D_in,H,D_out = 1000, 100, 10

N = 64 # batch_size


if th.cuda.is_available() :

	dtype = th.cuda.FloatTensor
else :
	dtype = th.Tensor
x = Variable(th.randn(N,D_in).type(dtype),requires_grad = False)
y = Variable(th.randn(N,D_out).type(dtype),requires_grad = False)


w1 = Variable(th.randn(D_in,H).type(dtype),requires_grad = True)
w2 = Variable(th.randn(H,D_out).type(dtype),requires_grad = True)

learning_rate = 1e-6 

model = nn.Sequential(nn.Linear(D_in,H) , nn.Sigmoid(), nn.Linear(H,D_out))

if th.cuda.is_available() :
	model.cuda()	

loss_fn = nn.MSELoss()


for t in range(200) :

	## Forward Pass 

	y_pred = model(x)

	loss = loss_fn(y_pred,y)

	print t,loss.data

	model.zero_grad()

	loss.backward()

	for params in model.parameters() :
		params.data -= params.grad.data * learning_rate



"""for t in range(no_of_batches) :


	## Forward Pass 

	h_input = x.dot(w1)
	## ReLU layer
	h_output  = h_input.clamp(min= 0)

	y_pred = h_output.dot(w2)


	## Computing loss here 
	loss = (y_pred - y).pow(2).sum()

	print t,loss.data

	w1.grad.data.zero_()
	w2.grad.data.zero_()

	loss.backward()

	w1.data -= learning_rate*w1.grad.data

	w2.data -= learning_rate*w2.grad.data
"""









