from numpy.core.fromnumeric import std
from PattRecClasses.GaussD import GaussD
import numpy as np
import scipy.stats
from .multigaussD import multigaussD
from .GaussD import GaussD

class GMM:
	def __init__(self, means, covs, weights):
		self.weights = weights
		self.means = means
		self.covs = covs
		self.n_dist = len(weights)
		#if self.means.shape[1] > 1:
		#self.Gauss_dist = [multigaussD(means[i], covs[i]) for i in np.arange(self.n_dist)]
		#else:
		self.Gauss_dist = [GaussD([means[i]], stdevs=[covs[i]]) for i in np.arange(self.n_dist)]

		self.dataSize = 1

	def rand(self, nData):
		data = np.array([dist.rand(nData)[0] for dist in self.Gauss_dist])
		return (self.weights@data)
		

	def likelihood(self, X):
		p = np.array([scipy.stats.multivariate_normal(self.means[i], self.covs[i], 1) for i in np.arange(len(self.means))])
		pdfs = np.array([p[gaussian].pdf(X[gaussian]) for gaussian in np.arange(len(p))])
		pdf = self.weights@pdfs

		return pdf

	def get_weights(self):
		return self.weights

	def get_pdfs(self,X):
		p = np.array([scipy.stats.multivariate_normal(self.means[i], self.covs[i], 1) for i in np.arange(len(self.means))])
		pdfs = np.array([p[gaussian].pdf(X[gaussian]) for gaussian in np.arange(len(p))])
		return pdfs

	def logprob(self, X):
		res = np.sum([self.Gauss_dist[i].logprob(np.array([X[i]]))  for i in np.arange(np.shape(X)[0])])
		return res

	