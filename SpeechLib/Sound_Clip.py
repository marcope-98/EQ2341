import numpy as np
from python_speech_features import base, sigproc
from scipy.io import wavfile
from scipy.fftpack import dct
from .speech_recon_utils import *

class Sound_Clip():


	def __init__(self, filename):
		self.filename =  filename
		self.samplerate, self.data = wavfile.read(filename)
		self.length   =  self.data.shape[0]/self.samplerate
		self.dtype    =  self.data.dtype 										# for now the class works with int16, but it could be expanded
		self.time     =  np.linspace(0.0, self.length, self.data.shape[0])



			
	def spectrogram(self, wintime=0.03, overlap=0.5, NFFT=None, winfunc=lambda x: np.ones((x,))):
		
		winpts  = int(np.ceil(    		wintime*self.samplerate))              							 # number of samples in the window
		winpts  = winpts  + ((winpts +1) % 2)																							 # make sure this is odd																							  
		steppts = int(np.ceil(overlap*wintime*self.samplerate))										    	   # number of samples overlapping
		steppts = steppts + ((steppts+1) % 2)																							 # make sure this is odd
		# if NFFT is not speciefied compute the next power of 2
		if NFFT == None:
			NFFT    = 2**(np.ceil(np.log(winpts)/np.log(2)))														 # fft padding, get next power of 2
			NFFT    = int(NFFT)                                             						 # make it an integer

		# divide signal into frames with overlap, and apply specified window
		frames_with_window = sigproc.framesig(self.data, winpts, steppts, winfunc=winfunc)
		# compute correction factor for applying the selected window
		Aw = 2/np.sum(winfunc(NFFT))                       							# amplitude correction factor for hanning window
		Ew = 2/(np.sum(np.square(winfunc(NFFT)))*self.samplerate)				# energy    correction factor for hanning window

		# compute power spectrum aka spectrogram (the scale is in decibel Full Scale dbFs)
		spect   = sigproc.magspec(frames_with_window,NFFT)							# magnitude spectrum (|STFT|)
		spect_2 = np.square(spect)*Ew																		# power 		spectrum (|STFT|^2)
		# compute the decibel full scale, the reference is the maximum int16 value of peak i.e. 32768
		log_spectro_dbfs = 10*np.log10(spect_2/32768**2)               # since we are using power the multiplicative factor is 10, 
																																	 #   the reference factor (denominator is the maximum value of amplitude which in terms of int16 data is 32768)

		# compute time and frequency arrays
		t = np.linspace(0, self.length,       num=np.shape(log_spectro_dbfs)[0]) # construct time
		f = np.linspace(0, self.samplerate/2, num=np.shape(log_spectro_dbfs)[1]) # construct frequencies

		return t, f, log_spectro_dbfs

	




	def mfcc(self, wintime=0.03,overlap=0.5, NFFT=None, nbands=40, nceps=13, lowfreq=133, highfreq=6854, winfunc=lambda x: np.ones((x,))):

		winpts  = int(np.ceil(    		wintime*self.samplerate))              							 # number of samples in the window
		winpts  = winpts  + ((winpts +1) % 2)																							 # make sure this is odd																							  
		steppts = int(np.ceil(overlap*wintime*self.samplerate))										    	   # number of samples overlapping
		steppts = steppts + ((steppts+1) % 2)																							 # make sure this is odd
		# if NFFT is not specified, compute next power of 2
		if NFFT==None:
			NFFT    = 2**(np.ceil(np.log(winpts)/np.log(2)))
			NFFT    = int(NFFT)
			
		# divide signal into frames with overlap, and apply specified window
		frames_with_window = sigproc.framesig(self.data, winpts, steppts, winfunc=winfunc)
		# compute correction factor for applying the selected window
		Aw =         2/np.sum(winfunc(NFFT))                       			# amplitude correction factor for hanning window
		Ew = 2/(np.sum(np.square(winfunc(NFFT)))*self.samplerate)				# energy    correction factor for hanning window

		# get stanley's auditory box filter bank
		fb = fbStanley(nbands, NFFT, self.samplerate, lowfreq, highfreq)
		# compute powerspectrum of signal and apply correction factor relative to selected window
		powspec = np.square(sigproc.magspec(frames_with_window,NFFT))*Ew	
		# apply filter bank to power spectrum
		fbank = np.dot(powspec, fb.T)
		# take the logarithm (the quantity are power therefore the multiplicative factor is 10)
		fbank = 10*np.log10(fbank)

		# apply the discrete cosine tranform
		mfccs = dct(fbank, type=2, axis=1, norm=None)
		# get only the first 13 cepstra
		mfccs = mfccs[:,:nceps]

		return mfccs


if __name__ == '__main__':
	pass