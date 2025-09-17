# generate variable binaural sounds - fluctuating frequency within range + small fluctuations on amplitude
# frequency oscilates within the wave range eg theta 4-8hz
# amplitudes oscilates slightly for low waves, more higher 

import numpy as np  
import random
from scipy.signal import butter,filtfilt, freqz as signal_freqz
import matplotlib.pyplot as plt # for testing 
import soundfile as sf



# generate randomized wave 
# waves values alloewed: 'alpha', 'beta', 'gamma', 'theta'
def gen_set(lod:list,sample_rate,seconds):
	# expected lod=[{'wave':'gamma', 'base_freq':220 } 
	# , {'wave':'gamma', 'base_freq':440 }]
		
	max_flu_time=60
	waves={'alpha':[8,12,30,0.98], 'beta':[12,30,10,0.8], 'gamma':[30,100,2,0.7], 'theta':[4,8,40,0.99]}	# f1,f2, min flu time 	, ampl max 
	
	# small adj: adjust sound amplitudes by frequency
	base_freq_ampl=[[100,0.8],[200,0.7],[400,0.6],[1000,0.5],[20000,0.4]]  
	def _get_base_ampl(xf ):
		ii=0
		while xf>base_freq_ampl[ii][0] and ii<len(base_freq_ampl) :
			ii+=1
		return base_freq_ampl[ii][1]
			
	stereo_audio = None
	sampl=int(seconds * sample_rate)
	t = np.linspace(0, seconds, sampl, False, dtype=np.float64)
	
	waves_phase={} # if same wave used on diff base freq - have same diff set not to add conflicting same waves 
	out_fname=f'sr_{sample_rate}'
	for di in lod:
	
		# first generate changes for wave and apply same if same wave is on another base !
		cur_wave=di['wave']
		out_fname=out_fname+'_'+cur_wave+'_'+str(di['base_freq'])
		
		sel_w=waves[ cur_wave ]
		fluct_time=[ sel_w[2], max_flu_time ] # to randomize - decide min and max tiem after which ampl or freq change can happen
		
		# min/max ampl per base freq - small adjustment
		ampl_max=_get_base_ampl( di['base_freq'])
		ampl_min=ampl_max*sel_w[3]		
		
		# randomize ampl fluctuations a bit 
		ampl_arr, arr_pos =_rand_steps(sampl,t,sample_rate,ampl_min,ampl_max,fluct_time[0], fluct_time[1] )
		
		# apply smooth ampl window start/end 
		w=window_onset_offset(t,sample_rate) # default smooth 5 s onset/offset start/end 
		ampl_arr[:len(w)]=ampl_arr[:len(w)]*w
		ampl_arr[-len(w):]=ampl_arr[-len(w):]*w[::-1]
		
		# BASE FREQ LEFT EAR  
		sin_left=ampl_min * np.sin(2 * np.pi *  di['base_freq'] *t )   
		
		# now rand part for frequency - to change between max/min of range per each 
		if  cur_wave not in waves_phase: 
			delta_freq_arr, _ =_rand_steps(sampl,t,sample_rate, sel_w[0],  sel_w[1] ,fluct_time[0], fluct_time[1] ) 
			waves_phase[ cur_wave ]=delta_freq_arr # later add with the main sine 
			# LATER REUSE for same WAVE on diff BASE FREQ
			
		sin_r= gen_variable_sine(sample_rate,   di['base_freq'] + waves_phase[cur_wave], ampl_arr  )	 
		
		if stereo_audio is None: # init signal 
			stereo_audio = np.array([sin_left , sin_r ])  
		else: # update signal 
			stereo_audio[0]=stereo_audio[0]+sin_left 
			stereo_audio[1]=stereo_audio[1]+ sin_r 
			
	if False:
		view_spectrum(sample_rate,t,stereo_audio )  
	else:			
		out_fname+='.mp3'
		# ensure scale sound to be < 0.95 < 1
		max_val = np.max(np.abs(stereo_audio))
		if max_val > 0.95:
			stereo_audio = stereo_audio * 0.95 / max_val
		sf.write(out_fname, stereo_audio.T, samplerate=sample_rate, bitrate_mode='VARIABLE', compression_level=0)	
		
		print(f"Created {out_fname}")

	
	
# gen sine with variable frequency - use cumulative phace calc 
def gen_variable_sine( sample_rate, freq_arr , ampl_arr ): 
	phase = 2 * np.pi * np.cumsum(freq_arr) / sample_rate  
	audio = ampl_arr * np.sin(phase) 	
	return audio 
	
	
	
def window_onset_offset(t_arr,sample_rate,grad_s=5): 
	_n=int(sample_rate*grad_s) 
	return (1-np.cos( np.pi *t_arr[:_n]/t_arr[_n] ) )/2



# create fluctuations within v1 v2 range with steps between min max flue time 
# + smooth out with filters 
def _rand_steps(sampl,t,sample_rate,v1,v2,min_flu_time, max_flu_time): 

	vinit=v1
	newpos=0
	arr_pos=[newpos]
	values=np.zeros(len(t),dtype=np.float64)
	next_v=0
	while True: 
		newpos2=newpos + int( random.uniform( min_flu_time, max_flu_time )* sample_rate ) 
		
		if newpos2>sampl:
			values[newpos:]=values[newpos-1]
			arr_pos.append(len(values)-1)
			break 
			
		next_v=random.uniform( v1, v2 )
		values[newpos:newpos2]=np.linspace(vinit, next_v, newpos2-newpos, False,dtype=np.float64) 
			
		newpos=newpos2
		vinit=next_v
		arr_pos.append(newpos)  
	
	# def _plot(x1,x2 ):
		# plt.subplot(2,1,1)
		# plt.plot(x1)
		# plt.subplot(2,1,2)
		# plt.plot(x2)
		# plt.show()		
		
	values2=low_pass_apply(values,samp_freq=sample_rate) #_plot(values,values2 ) 
	
	return values2, arr_pos



def low_pass_filter_set(samp_freq,cutoff=0.01, order=2, _test=False ): # 0.01Hz  
	b, a = butter( order, cutoff, btype='lowpass', analog=False, fs=samp_freq) 
	if _test: # test
		w, h = signal_freqz(b, a)
		w=w*samp_freq / (2 * np.pi)
		plt.semilogx(w, 20 * np.log10(abs(h)))
		plt.title('Butterworth filter frequency response')
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Amplitude [dB]') 
		plt.grid(which='both', axis='both')
		plt.axvline(cutoff, color='green') # cutoff frequency
		
		plt.show()
	
	
	return b,a
	
	

def low_pass_apply(y,samp_freq  ): # smooth random fluctuations
	# normalize before filtering
	va=	np.average(y) 
	vmin=np.min(y)
	vmax=np.max(y)
	_scale=(vmax-vmin) /2
	y_to_smooth= (y-va)/_scale
		
	b,a=low_pass_filter_set(samp_freq)
	y_low=filtfilt(b, a, y_to_smooth)*_scale+va
	
	# in case null/error ? none ?
	
	return y_low
	


def view_spectrum(fs,t,stereo_audio ): # for testing 

	left=stereo_audio[0]
	N = len(left)
	freq = np.fft.fftfreq(N, d=1/fs)  # Frequency axis in Hz
	fft_result_left = np.fft.fft(left)
	real_amplitudes_left = np.abs(fft_result_left.real) / N  # Normalize real part by signal length

	# Only plot positive frequencies (up to Nyquist)
	mask = freq >= 0
	freq = freq[mask]
	real_amplitudes_left = real_amplitudes_left[mask]
	
	fft_result_r = np.fft.fft(stereo_audio[1])
	real_amplitudes_r = np.abs(fft_result_r.real) / N  # Normalize real part by signal length
	real_amplitudes_r = real_amplitudes_r[mask]

	plt.subplot(2,1,1)
	plt.plot(freq, real_amplitudes_left)
	plt.title('Frequency Amplitudes (Real Part)')
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Amplitude') 
	plt.xlim(100,600)  # Limit to 1000 Hz for clarity
	
	plt.subplot(2,1,2)
	plt.plot(freq, real_amplitudes_r)
	plt.title('Frequency Amplitudes (Real Part)')
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Amplitude')
	plt.xlim(100,600)  # Limit to 1000 Hz for clarity
	
	plt.show()	
			
	
 	
if __name__ == "__main__":

	gamma=[#{'wave':'gamma', 'base_freq':220 } 
		 {'wave':'gamma', 'base_freq':440 }]
	
	beta=[#{'wave':'beta', 'base_freq':330 } 
		 {'wave':'beta', 'base_freq':440 }]	
		
	theta=[#{'wave':'theta', 'base_freq':330 } 
		 {'wave':'theta', 'base_freq':440 }]	
		
	alpha=[ #{'wave':'alpha', 'base_freq':330 } 
		  {'wave':'alpha', 'base_freq':440 }
		]	
		
	# bg=[{'wave':'beta', 'base_freq':220 }
		# , {'wave':'gamma', 'base_freq':440 } 
		# ]	
		
	seconds=600*3
	# sample_rate=22050*2 not really needed that much for <1khz sines
	sample_rate=22050//2 # should be more then enough, nq at 6khz...
	gen_set( alpha, sample_rate , seconds )
	
	
	
