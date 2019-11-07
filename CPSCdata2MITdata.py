import scipy.io as scio
import wfdb
from wfdb import processing
from scipy.io import loadmat
import sys
import os
import re
import numpy as np

def CPSC2MIT():
	data_path='mit-bih-arrhythmia-database-1.0.0/CPSC2019/data/'
	ref_path='mit-bih-arrhythmia-database-1.0.0/CPSC2019/ref/'
	for i in range(2000):
		print("processing round:",i)
		index="%05d" % (i+1)
		data_name=data_path+'data_'+index+'.mat'
		ref_name=ref_path+'R_'+index+'.mat'
		ecg_data=scio.loadmat(data_name)['ecg']
		ecg_ref_2d=scio.loadmat(ref_name)['R_peak']
		ecg_ref = list()
		symbols = list()
		for i in range(len(ecg_ref_2d)):
			ecg_ref.append(ecg_ref_2d[i][0])
			symbols.append('N')
		ecg_ref = np.array(ecg_ref)
		wfdb.wrsamp('CPSC'+index, fs=500, units=['mV'],
					sig_name=['I'], p_signal=ecg_data, fmt=['212'])
		wfdb.wrann('CPSC'+index, 'atr', ecg_ref, symbol=symbols)

if __name__=="__main__":
	CPSC2MIT()