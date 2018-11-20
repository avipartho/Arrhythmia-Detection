import scipy.io 
import pywt
import matplotlib.pyplot as pt
from scipy.signal import butter, lfilter
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

def time_to_seconds(timelist):
	secondslist = []
	for i in timelist:
		s = i.split(':')
		secondslist.append(float(s[0])*60+float(s[1]))
	return secondslist	

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	y = lfilter(b, a, data)
	return y

def features(data):
	# db2 DWT transformation
	a5,d5,d4,d3,d2,d1 = pywt.wavedec(data, 'db2', level=5)
	co_effs = [a5,d5,d4,d3,d2,d1]
	feature_list = []
	for i in co_effs:
		feature_list.append(np.mean(i))
		feature_list.append(np.std(i))
		feature_list.append(np.max(i))
		feature_list.append(np.min(i))
	return feature_list

def RpeakFromAtr(file_name):
	# beat_atr = ['N','L','R','B','A','a','J','S','V','r','F','e','j','n','E','/','f','Q','?']
	# beat_atr = {'N':0,'/':1,'L':2,'R':3,'V':4,'f':5,'A':6}
	beat_atr = {'N':0,'/':1,'L':2,'R':3,'V':4}
	atr_file = pd.read_csv('./Data/'+file_name)
	try: beats = atr_file['Seconds'].tolist()
	except: beats = time_to_seconds(atr_file['Time'].tolist())
	atrs = atr_file['Type'].tolist()
	all_beats,class_name = [],[]
	# checking for beat annotations from attribute file
	for count,i in enumerate(atrs):
		for j in beat_atr:
			if i == j:
				all_beats.append(float(format(beats[count],'0.3f')))
				class_name.append(beat_atr[j])
				break
	return all_beats,class_name

def prepare_dataset(pat_id, data, time_list):
	r_peaks,class_name = RpeakFromAtr(pat_id+'_atr.csv')
	file = open('Data.csv','a')
	y = open('y.csv','a')
	skipped = open('skipped_beat.csv','a')
	skipped.write(pat_id+',')
	for i in tqdm(r_peaks):
		index = time_list.index(i)
		try:
			samples = [data[i] for i in range(index-54,index+89)] #taking a total of 144 samples for 400 ms window range
			f = features(samples)	# extracting features
			# file.write(pat_id+',') # writing patient no.
			# file.write(str(i)+',') # writing rpeak time
			[file.write(str(i)+',') for i in f[:-1]] # writing all features
			file.write(str(f[-1])+'\n')
			y.write(str(class_name[r_peaks.index(i)])+'\n') # writing class name
		except:
			skipped.write(str(i)+',')
	skipped.write('\n')		
	file.close()
	skipped.close()

def load_data(pat_id, both_channel = False):
	d = scipy.io.loadmat('./Data/data/'+pat_id+'m.mat')
	data, data2, t = [], [], []
	
	for step,i in enumerate(d['val'][0]): #MLII data 
		data.append((i-1024)/200) # subtracting by bias and dividing by gain to convert from raw units to physical units 
		p = float(format(step*(1/360),'0.3f')) 
		t.append(p)	
		# t.append(step*(1/360))

	if(both_channel): 
		for step,i in enumerate(d['val'][1]): #V5 data
			data2.append((i-1024)/200)
		return data,data2,t
	else: return data,t

#.......loading data for MLII lead........
record_no =['100', '102', '103', '109', '118', '111', '208', '217', '221', '231', '233']
file = open('Data.csv','w')
skipped = open('skipped_beat.csv','w')
y = open('y.csv','w')
for i in record_no:
	ecg_data,time = load_data(i)

	#........data after bp filter/preprocessing.........

	# dataf = butter_bandpass_filter(ecg_data, 3, 20, 360, order=1) #2 timestamp right shift
	dataf = butter_bandpass_filter(ecg_data, 0.5, 45, 360, order=1) #1 timestamp right shift
	# dataf = butter_bandpass_filter(ecg_data, 0.1, 100, 360, order=1)

	# ...........features extraction...............

	prepare_dataset(i, dataf, time)

y.close()

# ............plotting..............

# ecg_data,ecg_data2,time = load_data('100', both_channel=True)
# dataf = butter_bandpass_filter(ecg_data, 0.5, 45, 360, order=1)
# dataf2 = butter_bandpass_filter(ecg_data2, 0.5, 45, 360, order=1)
# pt.plot(time[:360*2],ecg_data[:360*2],label='Original Signal(MLII)')
# pt.plot(time[:360*2],ecg_data2[:360*2],label='Original Signal(V5)')
# pt.plot(time[:360*2],dataf[:360*2],label='Filtered Signal(MLII) - 1st order(.5-45)')
# pt.plot(time[:360*2],dataf2[:360*2],label='Filtered Signal(V5) - 1st order(.5-45)')

# pt.xlabel('Time(s)')
# pt.ylabel('Volts(mV)')
# pt.title('ECG plot')
# pt.legend(loc = 1)

# pt.show()

print(Counter(np.loadtxt('y.csv'))) # printing collection of each class




