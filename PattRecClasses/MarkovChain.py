import numpy as np
from .DiscreteD import DiscreteD


class MarkovChain:
		"""
		MarkovChain - class for first-order discrete Markov chain,
		representing discrete random sequence of integer "state" numbers.
		
		A Markov state sequence S(t), t=1..T
		is determined by fixed initial probabilities P[S(1)=j], and
		fixed transition probabilities P[S(t) | S(t-1)]
		
		A Markov chain with FINITE duration has a special END state,
		coded as nStates+1.
		The sequence generation stops at S(T), if S(T+1)=(nStates+1)
		"""
		def __init__(self, initial_prob, transition_prob):

				self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
				self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]


				self.nStates = transition_prob.shape[0]

				self.is_finite = False
				if self.A.shape[0] != self.A.shape[1]:
						self.is_finite = True


		def probDuration(self, tmax):
				"""
				Probability mass of durations t=1...tMax, for a Markov Chain.
				Meaningful result only for finite-duration Markov Chain,
				as pD(:)== 0 for infinite-duration Markov Chain.
				
				Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
				"""
				pD = np.zeros(tmax)

				if self.is_finite:
						pSt = (np.eye(self.nStates)-self.A.T)@self.q

						for t in range(tmax):
								pD[t] = np.sum(pSt)
								pSt = self.A.T@pSt

				return pD

		def probStateDuration(self, tmax):
				"""
				Probability mass of state durations P[D=t], for t=1...tMax
				Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
				"""
				t = np.arange(tmax).reshape(1, -1)
				aii = np.diag(self.A).reshape(-1, 1)
				
				logpD = np.log(aii)*t+ np.log(1-aii)
				pD = np.exp(logpD)

				return pD

		def meanStateDuration(self):
				"""
				Expected value of number of time samples spent in each state
				"""
				return 1/(1-np.diag(self.A))
		
		def rand(self, tmax):
				"""
				S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
				
				Input:
				tmax= scalar defining maximum length of desired state sequence.
					 An infinite-duration MarkovChain always generates sequence of length=tmax
					 A finite-duration MarkovChain may return shorter sequence,
					 if END state was reached before tmax samples.
				
				Result:
				S= integer row vector with random state sequence,
					 NOT INCLUDING the END state,
					 even if encountered within tmax samples
				If mc has INFINITE duration,
					 length(S) == tmax
				If mc has FINITE duration,
					 length(S) <= tmaxs
				"""
				
				#*** Insert your own code here and remove the following error message
				
				# Initialize the random state sequence list with the first state
				#		Create a DiscreteD object with the initial probability and apply the method rand of the DiscreteD class
				#   the rhs will return a np.array of 1 element with the index of the first state
				S = DiscreteD(self.q).rand(1)

				# if the Markov Chain is finite (A.cols = A.rows + 1)
				if self.is_finite:
					# t = "current state (state 0 was calculated before the if statement)"
					t = 1
					# while loop (exit condition: t == tmax):
					#		the first state was t == 0
					#   we need to compute the next tmax - 1 states (if no other condition were to be applied)
					#   this can be achieved either with (while t <= tmax - 1) or (while t < tmax) 
					while t < tmax:
						# compute next state S_{t+1}
						#   self.A[ S[t-1] , :]		  -->  retrieve the row corresponding to the previous state
						#   DiscreteD(...).rand(1)  -->  construct DiscreteD object and apply rand method

						#		The rhs will return an np.array with one element (the index of the state t)
						state_tplus1 = DiscreteD(self.A[ S[t-1] , : ]).rand(1) 
						# exit condition: check if the state computed is the (additional) exit state
						if state_tplus1[0] == self.nStates:
							# return the State sequence computed till this point
							return S

						# otherwise add the S_{t+1} state to the state sequence
						S = np.concatenate((S, state_tplus1))
						# increment the current state variable
						t += 1
					# upon exiting the while loop return the state sequence
					return S
				# if the Markov Chain is infinite (A.cols == A.rows)
				else:
					# for loop starting from 1 until tmax (tmax not included) 
					for t in np.arange(1,tmax):
						# compute next state S_{t+1}
						state_tplus1 = DiscreteD(self.A[ S[t-1] , : ]).rand(1) 
						# add the state computed to the state sequence
						S = np.concatenate((S, state_tplus1))
					# upon exiting the for loop return the state sequence
					return S

		def viterbi(self):
				pass
		
		def stationaryProb(self):
				pass
		
		def stateEntropyRate(self):
				pass
		
		def setStationary(self):
				pass

		def logprob(self):
				pass

		def join(self):
				pass

		def initLeftRight(self):
				pass
		
		def initErgodic(self):
				pass

		def forward(self, pX):
			n_states = pX.shape[0]
			n_obs    = pX.shape[1]
			alpha_temp = np.empty(shape=(n_states, n_obs), dtype='float')
			alpha_hat  = np.empty(shape=(n_states, n_obs), dtype='float')
			c          = np.empty(shape=(n_obs),           dtype='float')
			# initialization
			alpha_temp[:,0] = self.q * pX[:,0]
			c[0]						= np.sum(alpha_temp[:,0])
			alpha_hat[:,0]  = alpha_temp[:,0]/c[0]
			# forward step
			for t in np.arange(1,np.shape(pX)[1]):
					alpha_temp[:,t] = pX[:,t]*np.dot(alpha_hat[:,t-1],self.A[:,:n_states])
					c[t]					  = np.sum(alpha_temp[:,t])
					alpha_hat[:,t]  = alpha_temp[:,t]/c[t]
			# termination
			if self.is_finite:
				c=np.append(c, np.dot(alpha_hat[:,-1],self.A[:,-1]))
			
			return alpha_hat, c


		def finiteDuration(self):
				pass
		
		def backwards(self, pX, c=None):
				if c is None:
					_, c = self.forward(pX)
				# preallacation:
				n_obs  	 = np.shape(pX)[1]
				n_states = np.shape(pX)[0]
				beta_hat = np.zeros(shape=(n_states,n_obs))

				c_alt  = np.flipud(c)
				pX_alt = np.fliplr(pX)
				
				# initialization:
				if (self.is_finite):
					beta_hat[:,0] = self.A[:,-1]/(c_alt[0]*c_alt[1])
				else:
					beta_hat[:,0] = 1/c_alt[0]

				# backward step
				for t in np.arange(1,n_obs):
					beta_hat[:,t] = (np.dot(self.A[:,:n_states], pX_alt[:,t-1]*beta_hat[:,t-1]))/c_alt[t+self.is_finite]

				return np.fliplr(beta_hat)
		
		def adaptStart(self):
				pass

		def adaptSet(self):
				pass

		def adaptAccum(self):
				pass

