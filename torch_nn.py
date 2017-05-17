import torch as th 
from torch.autograd import Variable
#dtype = th.cuda.Tensor



D_in ,H, D_out = 1000, 100, 10

batch_size = 64

x = Variable(th.cuda.randn(N,D_in),requires_grad = False)
y = Variable(th.cuda.randn(N,D_out),requires_grad = False)


w1 = Variable(th.cuda.randn(D_in,H),requires_grad = True)
w2 = Variable(th.cuda.randn(H,D_out),requires_grad = True)

learning_rate = 1e-6 

for t in range(no_of_batches) :


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










