import numpy as np

def CVMN(cepstra_array):
	res = cepstra_array - np.mean(cepstra_array, axis = 0)
	res = res / np.std(res, axis=0)
	return res


def mel2hz(m, lowfrequency=0):
	# check if m is a number or a list/numpy array
	if isinstance(m, np.ndarray):
		mels = m
	else:
		mels = np.asarray([m])

	linearSpacing = 200/3 																			# 2/3 * 100 = 66.666666..
	logSpacing = 1.0711703																			# exp(ln(f_c40/1000)/logFilters)  
																															#   where f_c40 = 6400 Hz (center frequency of the last logFilter)
	linearFilters = 13																					# from lowfrequency up to 1000 Hz
	logFilters = 27															
	totalFilters = linearFilters + logFilters										# 13 + 27 = 40

	Hz_1000 = 1000 # Hz
	mel_1000 = (Hz_1000-lowfrequency)/linearSpacing 						# last linear mel frequency
	Hz = []
	for mel in mels:
		if mel <= mel_1000:
			linear_coeff = lowfrequency + linearSpacing*mel					# linear spacing
			Hz.append(linear_coeff)
		else:
			log_coeff = Hz_1000*logSpacing**(mel-mel_1000)					# log spacing
			Hz.append(log_coeff)
			
	return np.array(Hz).T[0]


def hz2mel(f, lowfrequency=0):
	# check if m is a number or a list/numpy array
	if isinstance(f, np.ndarray):
		f_Hz = f
	else:
		f_Hz = np.asarray([f])
		
	linearSpacing = 200/3 																			# 2/3 * 100 = 66.666666..
	logSpacing = 1.0711703																			# exp(ln(f_c40/1000)/logFilters)  
																															#   where f_c40 = 6400 Hz (center frequency of the last logFilter)
	Hz_1000 = 1000 #Hz
	mel_1000 = (Hz_1000-lowfrequency)/linearSpacing							# last linear mel frequency

	mels = []
	for frequency in f_Hz:
		if frequency <= Hz_1000:
			linear_coeff = (frequency-lowfrequency)/linearSpacing									# linear spacing
			mels.append(linear_coeff)
		else:
			log_coeff = mel_1000 + np.log(frequency/Hz_1000)/np.log(logSpacing)		# log spacing
			mels.append(log_coeff)

	return np.array(mels)



def fbStanley(nfilt = 40, NFFT = 512, samplerate = 16000, lowfreq = 133, highfreq = 6854):
	f_min_mel = hz2mel(lowfreq,  lowfrequency=lowfreq)
	f_max_mel = hz2mel(highfreq, lowfrequency=lowfreq)
	mels = np.linspace(f_min_mel, f_max_mel, num=nfilt+2)
	mel_freqs = mel2hz(mels, lowfrequency=lowfreq)
	fftFreqs = np.arange(0,NFFT-1)*samplerate/NFFT
	indexes = np.arange(0, len(fftFreqs))
	
	fbank = np.zeros([nfilt,NFFT//2+1])
	for i in np.arange(0,nfilt):
		lower = indexes[(fftFreqs >= mel_freqs[i])   & (fftFreqs <= mel_freqs[i+1])]
		upper = indexes[(fftFreqs >= mel_freqs[i+1]) & (fftFreqs <= mel_freqs[i+2])]
		fbank[i,lower] = (fftFreqs[lower] - mel_freqs[i])   /(mel_freqs[i+1]-mel_freqs[i])   # k - f_bi-1 / f_bi  - f_bi-1
		fbank[i,upper] = (mel_freqs[i+2]  - fftFreqs[upper])/(mel_freqs[i+2]-mel_freqs[i+1]) # f_bi+1 - k / f_b+1 - f_bi

	enorm = 2.0 / (mel_freqs[2:nfilt+2] - mel_freqs[:nfilt])															 # Filterbank power = 1
	fbank *= enorm[:, np.newaxis]																													 # apply normalization factor

	return fbank