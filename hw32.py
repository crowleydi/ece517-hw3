import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 4 centroids of the gaussians
c=[[1,1],[2,1.5],[2,1],[3,1.5]]
N=10
sigma=0.2

# generate the X samples
X=np.zeros((N*len(c),2))
i = 0
for cent in c:
	for k in range(0,N):
		X[i,:]=cent+sigma*np.random.randn(1,2)
		i = i + 1
# group the first two and last two centroids
y=np.array([1]*(2*N)+[-1]*(2*N))

# scatter plot
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
plt.title(r"SVM with $\sigma$={}".format(sigma))
plt.show()

# fit the model
clf = svm.SVC(kernel='linear', C=100)
clf.fit(X, y)

# calculates weights as w = SV*a
# where SV are the support vectors and a are the dual coefficents
# from the classifier machine
w = np.matmul(clf.dual_coef_,clf.support_vectors_)
print(w.shape)

# get b from the model
b = clf.intercept_[0]

# calculate yhat, using our weights and b intercept
yhat = np.sign(np.matmul(w,X.T)+b)[0]
# compare yhat with the machines predicted value for y
# print the norm of the difference between the two vectors
print(np.linalg.norm(clf.predict(X)-yhat))

#
# plot the decision function
#
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
# the decision function determines where -1, 0, and 1 are
# on the plot.
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# plot support vectors
# draws circles around the points in the previous scatter plot
# which are the support vector points
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.title(r"SVM with $\sigma$={}".format(sigma))
plt.show()
