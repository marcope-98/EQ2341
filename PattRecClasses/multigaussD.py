import numpy as np
import scipy.stats

class multigaussD:
		mean = np.array([0])
		cov = np.array([[0]])
		
		
		def __init__(self, mu, C):
				if C.shape[0] is not C.shape[1]:
						print("error, non-square covariance matrix supplied")
						return
				if mu.shape[0] is not C.shape[0]:
						print("error, mismatched mean vector and covariance matrix dimensions")
						return
				self.mean = mu
				if np.where(np.diag(C)==0)[0].shape[0] != 0:
						C += np.diagflat(np.ones(C.shape[0])/10000)
				C[np.isnan(C)]=1
				self.cov = C
				
				self.dataSize = len(self.mean)
				return
		
		def random(self, num):
				return np.random.multivariate_normal(self.mean, self.cov, num)
		
		def rand(self,num):
				return np.random.multivariate_normal(self.mean, self.cov, 1)[0]
		
		def likelihood(self, X):
				p = scipy.stats.multivariate_normal(self.mean, self.cov, 1)
				pd = p.pdf(X)
				return pd
		
		def loghood(self, X):
				return np.log(self.likelihood(X))
		def getmean(self):
				return self.mean
		def getcov(self):
				return self.cov