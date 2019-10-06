import numpy as np
import numpy.matlib as matlib
import matplotlib.pylab as plt
from tqdm import tqdm
from sklearn import svm

def data(N,sigma):
	w = np.ones(10)/np.sqrt(10)
	w1 = [1., 1., 1., 1., 1., -1., -1., -1., -1., -1.]/np.sqrt(10)
	w2 = [-1., -1., 0, 1., 1., -1., -1., 0, 1., 1.]/np.sqrt(8)
	
	x = np.zeros((4,10))
	x[1,:] = x[0,:] + sigma*w1
	x[2,:] = x[0,:] + sigma*w2
	x[3,:] = x[2,:] + sigma*w1
	
	X1 = x + sigma*matlib.repmat(w,4,1)/2
	X2 = x - sigma*matlib.repmat(w,4,1)/2

	X1 = matlib.repmat(X1, 2*N, 1)
	X2 = matlib.repmat(X2, 2*N, 1)

	X = np.concatenate((X1,X2), axis = 0)
	X = X + 0.2*sigma*np.random.randn(16*N,10)
	Y = np.array([1]*(8*N)+[-1]*(8*N))
	Z = np.random.permutation(16*N)

	Z = Z[:N]
	X = X[Z,:]
	Y = Y[Z]
	
	return X, Y


def calcError(sigma, N, C, iterations):
	empErr = 0
	testErr = 0
	for i in range(0,iterations):
		X,y = data(N,sigma)
		clf = svm.SVC(C=C, kernel='linear', gamma='scale').fit(X, y)
		e = clf.predict(X)-y
		empErr = empErr + np.sum(e*e)/len(y)
		Xtest,ytest = data(N,sigma)
		e = clf.predict(Xtest)-ytest
		testErr = testErr + np.sum(e*e)/len(ytest)

	return empErr/iterations, testErr/iterations

sigma = 1

if True:
	C = 1
	N = np.linspace(10,500,50,dtype=int)
	etest = np.zeros(len(N))
	eemp = np.zeros(len(N))
	for k in tqdm(range(len(N))):
		eemp[k], etest[k] = calcError(sigma, N[k], C, 500)
	
	plt.plot(N,etest,label='Actual')
	plt.plot(N,eemp,label='Empirical')
	plt.xlabel("N")
	plt.legend()
	plt.title(r'Risk vs. Num samples for $C={},\sigma={}$'.format(C,sigma))
	plt.show()

if True:
	N = 100
	C = 10**np.linspace(-1.5,1,100)
	etest = np.zeros(len(C))
	eemp = np.zeros(len(C))
	for k in tqdm(range(len(C))):
		eemp[k], etest[k] = calcError(sigma, N, C[k], 1000)
	
	plt.plot(C,etest,label='Actual')
	plt.plot(C,eemp,label='Empirical')
	plt.plot(C,(etest-eemp),label='Structural')
	plt.xscale(value="log")
	plt.xlabel("C")
	plt.legend()
	plt.title(r'Risk vs. Complexity for $N={},\sigma={}$'.format(N,sigma))
	plt.show()

