'''
We will use the python_speech_features library for 
extracting the MFCC components for speech sounds.

You can find the installation instructions and other
ïnformation regarding this library here:

https://pypi.org/project/python_speech_features/0.4/

This library also supports additional speech features 
beyond MFCC. You are allowed to use such features if you
find them interesting/relevant for the project task. 
However, consider such features strictly optional and
an add-on to the MFCC. If you end up using more feaures,
please make sure to highlight them clearly in your final 
report and presentation. Good luck!
'''

from python_speech_features import base, sigproc
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import dct


def GetSpeechFeatures(signal, fs):
	# this was not used in the python notebook we retrieve the cepstra coefficient using the compute_cepstrogram function:
		# moreover in our implementation we did not use any lifter or preemphasis.
		# the filter bank was computed using the FB-40 proposed by Stanley's auditory toolbox, which achieves the minimum EER across other filter banks
		features_mfcc = base.mfcc(signal, fs,0.03)
		
		return features_mfcc

def mel2hz(mels, lowfrequency=0):
	linearSpacing = 200/3 					# 2/3 * 100 = 66.666666..
	logSpacing = 1.0711703					
	linearFilters = 13							# from lowfrequency up to 1000 Hz
	Hz_1000 = 1000 # Hz
	logFilters = 27

	totalFilters = linearFilters + logFilters
	
	mel_1000 = (1000-lowfrequency)/linearSpacing # last linear mel frequency
	Hz = []
	for mel in mels:
		if mel <= mel_1000:
			linear_coeff = lowfrequency + linearSpacing*mel					# linear spacing
			Hz.append(linear_coeff)
		else:
			log_coeff = Hz_1000*logSpacing**(mel-mel_1000)					# log spacing
			Hz.append(log_coeff)
			
	return np.array(Hz).T

def hz2mel(f_Hz, lowfrequency=0):
	linearSpacing = 200/3 
	logSpacing = 1.0711703
	Hz_1000 = 1000 #Hz


	mel_1000 = (1000-lowfrequency)/linearSpacing	# last linear mel frequency
	mels = []
	for frequency in f_Hz:
		if frequency <= Hz_1000:
			linear_coeff = (frequency-lowfrequency)/linearSpacing						# linear spacing
			mels.append(linear_coeff)
		else:
			log_coeff = mel_1000 + np.log(frequency/Hz_1000)/np.log(logSpacing)		# log spacing
			mels.append(log_coeff)

	return np.array(mels).T

def compute_spectrogram(filename):
	# read from file
	samplerate, data = wavfile.read(filename)
	time_window = 0.03 #s
	length = data.shape[0]/samplerate
	winpts  = int(np.ceil(    time_window*samplerate))              # number of samples in the window
	steppts = int(np.ceil(0.5*time_window*samplerate))			    		# number of samples overlapping
	NFFT    = 2**(np.ceil(np.log(winpts)/np.log(2)))								# fft padding, get next power of 2
	NFFT    = int(NFFT)                                             # make it an integer

	# divide signal into frames with 50% overlap, and apply hanning window
	frames_with_window = sigproc.framesig(data, winpts, steppts, winfunc=np.hanning)
	# compute correction factor for applying the hanning window
	Aw =         2/np.sum(np.hanning(NFFT))                       # amplitude correction factor for hanning window

	# since we are interested in the first half of the fourier transform (before Nyquist frequency, we multiply by 2 and apply the correction factor)
	Ew = 2/(np.sum(np.square(np.hanning(NFFT)))*samplerate)				# energy    correction factor for hanning window

	# compute power spectrum aka spectrogram (the scale is in decibel Full Scale dbFs)
	spectrogram_ = sigproc.magspec(frames_with_window,NFFT)    # multiply by 2 to compensate the spectrum over Nyquist frequency and apply energy/power correction factor
	spectrogram_squared = np.square(spectrogram_)*Ew
	# compute the decibel full scale, the reference is the maximum int16 value of peak i.e. 32768
	log_spectro_dbfs = 10*np.log10(spectrogram_squared/32768**2)               # since we are using power the multiplicative factor is 10, the reference factor (denominator is the maximum value of amplitude which in terms of int16 data is 32768)

	# compute time and frequency arrays
	t = np.linspace(0, length,      num=np.shape(log_spectro_dbfs)[0]) # construct time
	f = np.linspace(0, samplerate/2,num=np.shape(log_spectro_dbfs)[1]) # construct frequencies
	return t, f, log_spectro_dbfs

def compute_cepstrogram(filename):
	# read from file
	samplerate, data = wavfile.read(filename)
	time_window = 0.03 #s

	winpts  = int(np.ceil(    time_window*samplerate))              # number of samples in the window
	steppts = int(np.ceil(0.5*time_window*samplerate))			    		# number of samples overlapping
	NFFT    = 2**(np.ceil(np.log(winpts)/np.log(2)))								# fft padding, get next power of 2
	NFFT    = int(NFFT) 

	# create 50% overlapping frames with hanning window
	frames_with_window = sigproc.framesig(data, winpts, steppts, winfunc=np.hanning)
	# compute correction factor for applying the hanning window
	Aw =         2/np.sum(np.hanning(NFFT))                       # amplitude correction factor for hanning window

	Ew = 2/(np.sum(np.square(np.hanning(NFFT)))*samplerate)				# energy    correction factor for hanning window

	
	nfiltb = 30                                     # number of bands of the filter 
	nceps = 13                                      # number of cepstra (we are interested in the first 13)
	lowfreq = 20                                    # low frequency
	highfreq = 4000                                 # high frequency

	fb = base.get_filterbanks(nfiltb, NFFT,samplerate,lowfreq,highfreq)             # get filterbank
	# ----------------------------------------------------------------------------------------------------
	# this code is the same as fbank in python_speech_features (we changed the conversion in accordance to the filterbank fb-40 proposed by Slaney's auditory toolbox)
	f_min_mel = hz2mel([lowfreq])

	f_max_mel = hz2mel([highfreq])

	mels = np.linspace(f_min_mel, f_max_mel, num=nfiltb+2)			# the mels are linearly spaced
	mel_freqs = mel2hz(mels)[0]
	# ----------------------------------------------------------------------------------------------------
	# normalize energy of filter so it has energy = 1
	# for each traingular filter in the filter bank, get the higher and lower frequency where the weight is non zero
	enorm = 2.0 / (mel_freqs[2:nfiltb+2] - mel_freqs[:nfiltb])
	fb *= enorm[:, np.newaxis]											# apply normalization factor
	
	plt.scatter(np.arange(len(fb[0,fb[0]>0])),fb[0,fb[0]>0])
	plt.show()
	powspec = np.square(sigproc.magspec(frames_with_window,NFFT))*Ew	# compute powerspectrum of signal and apply correction factor relative to hann window
	# apply filter bank to power spectrum
	fbank = np.dot(powspec, fb.T)										# apply filter bank
	# take the logarithm (the quantity are power therefore the multiplicative factor is 10)
	fbank = 10*np.log10(fbank)

	# apply the discrete cosine tranform
	mfcc = dct(fbank,type=2,axis=1,norm=None)
	# get only the first 13 cepstra
	mfcc = mfcc[:,:nceps]

	return mfcc


if __name__ == '__main__':
	compute_cepstrogram('./Sounds/female.wav')