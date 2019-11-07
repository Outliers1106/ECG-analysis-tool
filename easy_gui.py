import easygui as g
import wfdb
import numpy as np
from wfdb import processing
from scipy.io import loadmat
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import sys
import os
import ctypes
import _ctypes
import re
import win32api
class __Autonomy__(object):
    """
    自定义变量的write方法
    """
    def __init__(self):
        """
        init
        """
        self._buff = ""

    def write(self, out_stream):
        """
        :param out_stream:
        :return:
        """
        self._buff += out_stream

def samp2time(samples,freq):
    seconds=samples/freq
    second=seconds%60
    second=round(second,3)
    seconds=int(seconds/60)
    min=seconds%60
    seconds=int(seconds/60)
    hour=seconds
    return str(hour)+':'+str(min)+':'+str(second)


def draw_graph(r_peak_inds,sig,fields,algorithm_name,qrs_inds=None,p_inds=None,t_inds=None):
	print(qrs_inds)
	if len(qrs_inds)==1:
		num=qrs_inds[0]
		qrs_inds=np.array([-100])
		qrs_inds=np.append(qrs_inds,num)
	print(qrs_inds)
	#至少有两个点才能用下面这个函数
	comparitor = processing.compare_annotations(ref_sample=r_peak_inds,  # 真实的r波位置
												test_sample=qrs_inds,  # 算法算出的r波位置
												window_width=int(0.1 * fields['fs']),
												signal=sig[:, 0])  # 信号
	# 将comparitor.print_summary()输出的内容重定向到a._buffer中
	current = sys.stdout
	a = __Autonomy__()
	# 会调用a的write方法, 和self._buff的内容拼接
	sys.stdout = a
	comparitor.print_summary()
	sys.stdout = current
	# 输出捕获的内容
	# print("捕获内容", a._buff)
	summary = a._buff
	summary = summary.replace("\n", "<br>")
	summary = summary.split("<br>")
	# 将summary内容分段成列表
	# plt.rcParams['figure.figsize']=(16,9)
	# plt.rcParams['savefig.dpi']=300
	# plt.rcParams['figure.dpi']=300
	fig, ax = comparitor.plot(title=algorithm_name+' detected QRS vs reference annotations', return_fig=True)
	# figure 保存为二进制文件
	# buffer = BytesIO()
	# plt.savefig(buffer)
	#标注未识别和识别错误的点
	unmatched_ref_sample = comparitor.unmatched_ref_sample
	unmatched_test_sample = comparitor.unmatched_test_sample
	for sample in unmatched_ref_sample:
		ax.scatter([sample, ], [sig[sample][0], ], 50, color='red')
		ax.annotate(r'unmatched_ref',
				 xy=(sample, sig[sample][0]), xycoords='data',
				 xytext=(+0, +15), textcoords='offset points', fontsize=8,
				 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
	for sample in unmatched_test_sample:
		ax.scatter([sample, ], [sig[sample][0], ], 50, color='red')
		ax.annotate(r'unmatched_test',
				 xy=(sample, sig[sample][0]), xycoords='data',
				 xytext=(+0, +15), textcoords='offset points', fontsize=8,
				 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

	if algorithm_name=='EcgAnalysis_Algorithm and HRV_Analysis':

		pass
	plt.grid()
	plt.show(ax)
	# plotdata = buffer.getvalue()
	return summary
#读取MIT数据，返回r波峰值位置
#文件名 开始点 结束点
#'mit-bih-arrhythmia-database-1.0.0/100' 0 10000
def read_r_peak(filename,sampfrom=None,sampto=None):
	annotation=wfdb.rdann(filename,'atr',sampfrom=sampfrom,sampto=sampto)
	sample=annotation.sample
	symbol=annotation.symbol
	#删除非r波的标注
	AAMI_MIT_MAP = {'N': 'Nfe/jnBLR',  # 将19类信号分为五大类，这19类与r波位置相关
					'S': 'SAJa',
					'V': 'VEr',
					'F': 'F',
					'Q': 'Q?'}
	MIT2AAMI = {c: k for k in AAMI_MIT_MAP.keys() for c in AAMI_MIT_MAP[k]}
	mit_beat_codes=list(MIT2AAMI.keys())
	symbol=np.array(symbol)
	print(symbol)
	isin=np.isin(symbol,mit_beat_codes)
	sample=sample[isin]
	return sample

#文件名 开始点 结束点 通道 r波坐标
#'mit-bih-arrhythmia-database-1.0.0/100' 0 10000 0或1 list
def xqrs_algorithm(filename,sampfrom=None,sampto=None,channel=0,r_peak_inds=None):
	sig, fields = wfdb.rdsamp(filename, channels=[channel],sampfrom=sampfrom,sampto=sampto)

	qrs_inds=processing.xqrs_detect(sig=sig[:,0],fs=fields['fs'])#numpy.array
	print(qrs_inds)
	#标记位置减去采样点开始位置得到相对位置
	r_peak_inds-=sampfrom
	summary=draw_graph(r_peak_inds, sig, fields, 'XQRS',qrs_inds)

	return summary

#文件名 开始点 结束点 通道 r波坐标
#'mit-bih-arrhythmia-database-1.0.0/100' 0 10000 0或1 list
def gqrs_algorithm(filename,sampfrom=None,sampto=None,channel=0,r_peak_inds=None):
	sig, fields = wfdb.rdsamp(filename, channels=[channel], sampfrom=sampfrom,sampto=sampto)
	qrs_inds=processing.gqrs_detect(sig=sig[:,0],fs=fields['fs'])
	# 标记位置减去采样点开始位置得到相对位置
	r_peak_inds -= sampfrom
	summary =draw_graph(r_peak_inds, sig, fields, 'GQRS', qrs_inds)

	return summary


def wqrs_algorithm(ecg_file_name,sampfrom=None,sampto=None,channel=0,r_peak_inds=None):
	sig, fields = wfdb.rdsamp(ecg_file_name, channels=[channel], sampfrom=sampfrom, sampto=sampto)
	r_peak_inds -= sampfrom

	ecg_file_name_list = ecg_file_name.split('/')
	ecg_name = ecg_file_name_list[len(ecg_file_name_list) - 1]

	argv1=[
		"wqrs","-r",ecg_name,"-f","s"+str(sampfrom),"-t","s"+str(sampto),"-s",str(channel)
	]

	argv2=[
		"rdann","-a","wqrs","-r",ecg_name
	]
	argc1=len(argv1)
	argc2=len(argv2)
	MAX_STR_LEN=20
	#参数参考 https://archive.physionet.org/physiotools/wag/wqrs-1.htm
	c_char_data1 = (ctypes.c_char_p * len(argv1))()
	c_char_data2 = (ctypes.c_char_p * len(argv2))()
	for i in range(len(argv1)):
		c_char = (ctypes.c_char * MAX_STR_LEN)()
		c_char.value = argv1[i].encode('utf-8')
		c_char_data1[i] = ctypes.c_char_p(c_char.value)
	for i in range(len(argv2)):
		c_char = (ctypes.c_char * MAX_STR_LEN)()
		c_char.value = argv2[i].encode('utf-8')
		c_char_data2[i] = ctypes.c_char_p(c_char.value)
	#进行类型转换 c语言原型(int, char**, int, char**)
	BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	print(BASE_DIR)
	path_dll = os.path.join(BASE_DIR, 'ECG-analysis-tool/DLL').replace("\\","/")
	os.environ['path'] += ';' + path_dll  # 添加依赖文件目录 即 D:\ECG_PROJECT\yihaecg-web\dashboard\DLL
	#print(os.environ['path'])
	#print(path_dll)
	#print(os.getcwd())
	#path_dll=path_dll+'/Project_qrs.dll'
	#dll = CDLL(path_dll)
	dll=ctypes.cdll.LoadLibrary('Project_qrs.dll')

	dll.wqrs_func.restype = ctypes.POINTER(ctypes.c_int)#函数返回值为int* 需要如此转换
	res=dll.wqrs_func(argc1,c_char_data1,argc2,c_char_data2)
	print(dll)

	_ctypes.FreeLibrary(dll._handle)
	print(dll)
	os.environ['path'].strip(';'+path_dll)
	#win32api.FreeLibrary(dll._handle)

	qrs_inds_len=res[0]#返回int数组 在第0位记录数组长度
	if qrs_inds_len<0:
		qrs_inds= np.array([-100])
		summary = draw_graph(r_peak_inds, sig, fields, 'WQRS', qrs_inds)
		return summary

	qrs_inds=list()
	for i in range(qrs_inds_len):
		if res[i+1]>=sampfrom and res[i+1]<=sampto:
			qrs_inds.append(res[i+1])
	qrs_inds=np.array(qrs_inds)
	print(qrs_inds)
	qrs_inds-=sampfrom


	summary =draw_graph(r_peak_inds, sig, fields, 'WQRS', qrs_inds)

	return summary


def sqrs_algorithm(ecg_file_name,sampfrom=None,sampto=None,channel=0,r_peak_inds=None,freq=None):

	ecg_file_name_list = ecg_file_name.split('/')
	ecg_name = ecg_file_name_list[len(ecg_file_name_list) - 1]
	sig, fields = wfdb.rdsamp(ecg_file_name, channels=[channel], sampfrom=sampfrom, sampto=sampto)
	r_peak_inds -= sampfrom

	argv1=[
		"sqrs","-r",ecg_name,"-f",samp2time(sampfrom,freq),"-t",samp2time(sampto,freq),"-s",str(channel)
	]

	argv2=[
		"rdann","-a","sqrs","-r",ecg_name
	]
	argc1=len(argv1)
	argc2=len(argv2)
	MAX_STR_LEN=20
	#参数参考 https://archive.physionet.org/physiotools/wag/wqrs-1.htm
	c_char_data1 = (ctypes.c_char_p * len(argv1))()
	c_char_data2 = (ctypes.c_char_p * len(argv2))()
	for i in range(len(argv1)):
		c_char = (ctypes.c_char * MAX_STR_LEN)()
		c_char.value = argv1[i].encode('utf-8')
		c_char_data1[i] = ctypes.c_char_p(c_char.value)
	for i in range(len(argv2)):
		c_char = (ctypes.c_char * MAX_STR_LEN)()
		c_char.value = argv2[i].encode('utf-8')
		c_char_data2[i] = ctypes.c_char_p(c_char.value)
	#进行类型转换 c语言原型(int, char**, int, char**)
	BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	print(BASE_DIR)
	path_dll = os.path.join(BASE_DIR, 'ECG-analysis-tool/DLL').replace("\\","/")
	os.environ['path'] += ';' + path_dll  # 添加依赖文件目录 即 D:\ECG_PROJECT\yihaecg-web\dashboard\DLL
	#print(os.environ['path'])
	#print(path_dll)
	#print(os.getcwd())
	#path_dll=path_dll+'/Project_qrs.dll'
	#dll = CDLL(path_dll)
	dll=ctypes.cdll.LoadLibrary('Project_qrs.dll')

	dll.sqrs_func.restype = ctypes.POINTER(ctypes.c_int)#函数返回值为int* 需要如此转换
	res=dll.sqrs_func(argc1,c_char_data1,argc2,c_char_data2)
	print(dll)

	_ctypes.FreeLibrary(dll._handle)
	print(res)
	os.environ['path'].strip(';'+path_dll)
	#win32api.FreeLibrary(dll._handle)

	qrs_inds_len=res[0]#返回int数组 在第0位记录数组长度
	if qrs_inds_len<0:
		qrs_inds= np.array([-100])
		summary = draw_graph(r_peak_inds, sig, fields, 'SQRS', qrs_inds)
		return summary

	qrs_inds=list()
	for i in range(qrs_inds_len):
		if res[i+1]<=sampto and res[i+1]>=sampfrom:
			qrs_inds.append(res[i+1])
	qrs_inds=np.array(qrs_inds)
	print(qrs_inds)
	qrs_inds -= sampfrom
	# 标记位置减去采样点开始位置得到相对位置
	summary =draw_graph(r_peak_inds, sig, fields, 'SQRS', qrs_inds)

	return summary

def sqrs125_algorithm(ecg_file_name,sampfrom=None,sampto=None,channel=0,r_peak_inds=None,freq=None):
	sig, fields = wfdb.rdsamp(ecg_file_name, channels=[channel], sampfrom=sampfrom, sampto=sampto)
	r_peak_inds -= sampfrom
	ecg_file_name_list = ecg_file_name.split('/')
	ecg_name = ecg_file_name_list[len(ecg_file_name_list) - 1]
	argv1=[
		"125sqrs","-r",ecg_name,"-f",samp2time(sampfrom,freq),"-t",samp2time(sampto,freq),"-s",str(channel)
	]

	argv2=[
		"rdann","-a","125sqrs","-r",ecg_name
	]
	argc1=len(argv1)
	argc2=len(argv2)
	MAX_STR_LEN=20
	#参数参考 https://archive.physionet.org/physiotools/wag/wqrs-1.htm
	c_char_data1 = (ctypes.c_char_p * len(argv1))()
	c_char_data2 = (ctypes.c_char_p * len(argv2))()
	for i in range(len(argv1)):
		c_char = (ctypes.c_char * MAX_STR_LEN)()
		c_char.value = argv1[i].encode('utf-8')
		c_char_data1[i] = ctypes.c_char_p(c_char.value)
	for i in range(len(argv2)):
		c_char = (ctypes.c_char * MAX_STR_LEN)()
		c_char.value = argv2[i].encode('utf-8')
		c_char_data2[i] = ctypes.c_char_p(c_char.value)
	#进行类型转换 c语言原型(int, char**, int, char**)
	BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	print(BASE_DIR)
	path_dll = os.path.join(BASE_DIR, 'ECG-analysis-tool/DLL').replace("\\","/")
	os.environ['path'] += ';' + path_dll  # 添加依赖文件目录 即 D:\ECG_PROJECT\yihaecg-web\dashboard\DLL
	#print(os.environ['path'])
	#print(path_dll)
	#print(os.getcwd())
	#path_dll=path_dll+'/Project_qrs.dll'
	#dll = CDLL(path_dll)
	dll=ctypes.cdll.LoadLibrary('Project_qrs.dll')

	dll.sqrs125_func.restype = ctypes.POINTER(ctypes.c_int)#函数返回值为int* 需要如此转换
	res=dll.sqrs125_func(argc1,c_char_data1,argc2,c_char_data2)
	print(dll)

	_ctypes.FreeLibrary(dll._handle)
	print(res)
	os.environ['path'].strip(';'+path_dll)
	#win32api.FreeLibrary(dll._handle)

	qrs_inds_len=res[0]#返回int数组 在第0位记录数组长度
	if qrs_inds_len<0:
		qrs_inds= np.array([-100])
		summary = draw_graph(r_peak_inds, sig, fields, 'SQRS125', qrs_inds)
		return summary

	qrs_inds=list()
	for i in range(qrs_inds_len):
		if res[i+1]<=sampto and res[i+1]>=sampfrom:
			qrs_inds.append(res[i+1])
	qrs_inds=np.array(qrs_inds)
	print(qrs_inds)
	# 标记位置减去采样点开始位置得到相对位置
	qrs_inds -= sampfrom

	summary =draw_graph(r_peak_inds, sig, fields, 'SQRS125', qrs_inds)

	return summary

def EcgAnalysis_algorithm(ecg_file_name,sampfrom=0,sampto=None,channel=0,r_peak_inds=None):
	ecg_file_name_list = ecg_file_name.split('/')
	ecg_name = ecg_file_name_list[len(ecg_file_name_list) - 1]
	BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	print(BASE_DIR)
	path_dll = os.path.join(BASE_DIR, 'ECG-analysis-tool/DLL').replace("\\","/")
	os.environ['path'] += ';' + path_dll  # 添加依赖文件目录 即 D:\ECG_PROJECT\yihaecg-web\dashboard\DLL
	dll=ctypes.cdll.LoadLibrary('Project_qrs.dll')
	#类型转换
	argv=ctypes.c_char_p(ecg_name.encode('utf-8'))
	dll.EcgAnalysis_func(argv,sampfrom,sampto)
	_ctypes.FreeLibrary(dll._handle)
	os.environ['path'].strip(';'+path_dll)

	#EcgAnalysis_func会在mit目录中生成对应的数据，现在读取所需要的数据

	qrs_inds,p_inds,t_inds=read_peaks_and_intervals_for_EcgAnalysis(sampfrom,sampto)

	#读取HRV各参数
	f=open('foo.nnstat')
	summary_hrv=list()
	for line in open('foo.nnstat'):
		line=f.readline()
		summary_hrv.append(line)
	f.close()
	f=open('foo.pwr')
	for line in open('foo.pwr'):
		line=f.readline()
		summary_hrv.append(line)
	f.close()


	# 标记位置减去采样点开始位置得到相对位置
	r_peak_inds -= sampfrom
	qrs_inds -= sampfrom
	p_inds-= sampfrom
	t_inds-= sampfrom


	sig, fields = wfdb.rdsamp(ecg_file_name, channels=[channel], sampfrom=sampfrom, sampto=sampto)
	summary=draw_graph(r_peak_inds, sig, fields, 'EcgAnalysis', qrs_inds)
	# 将HRV参数结果放入summary中
	for item in summary_hrv:
		summary.append(item)

	return summary

def read_peaks_and_intervals_for_EcgAnalysis(sampfrom,sampto):
	f = open('mit-bih-arrhythmia-database-1.0.0/QRSbt.txt')  # 即Ramp.txt，都一样
	qrs_inds = list()
	for line in open('mit-bih-arrhythmia-database-1.0.0/QRSbt.txt'):
		line = f.readline()
		if line == "":
			break
		line = line.split(" ")
		num = int(line[1])
		if num > sampto:
			break
		if num < sampfrom:
			continue
		qrs_inds.append(num)
	f.close()
	qrs_inds = np.array(qrs_inds)

	f = open('mit-bih-arrhythmia-database-1.0.0/Pamp.txt')
	p_inds = list()
	for line in open('mit-bih-arrhythmia-database-1.0.0/Pamp.txt'):
		line = f.readline()
		if line == "":
			break
		line = line.split(" ")
		while line.count("")>0:
			line.remove("")
		num = int(re.findall(r'\d+',line[2])[0])
		if num > sampto:
			break
		if num < sampfrom:
			continue
		p_inds.append(num)
	f.close()
	p_inds = np.array(p_inds)

	f = open('mit-bih-arrhythmia-database-1.0.0/Tamp.txt')
	t_inds = list()
	for line in open('mit-bih-arrhythmia-database-1.0.0/Tamp.txt'):
		line = f.readline()
		if line == "":
			break
		line = line.split(" ")
		while line.count("") > 0:
			line.remove("")
		num = int(re.findall(r'\d+', line[2])[0])
		if num > sampto:
			break
		if num < sampfrom:
			continue
		t_inds.append(num)
	f.close()
	t_inds = np.array(t_inds)
	return qrs_inds,p_inds,t_inds


def select_one_record():
	pass

def select_mult_record():
	pass

def select_one_algorithm():
	pass

def select_mult_algorithm():
	pass

while True:
	#part 1 选择数据
	choices_list=[]
	for i in range(2000):
		num="%05d" % (i+1)
		choices_list.append('CPSC'+num)
	for i in range(25):
		choices_list.append(str(100+i))
	for i in range(35):
		choices_list.append(str(200+i))

	msg="请选择心电数据"
	title="选择心电数据"
	ecg_name=g.multchoicebox(msg,title,choices=choices_list)
	ecg_name=ecg_name[0]

	if ecg_name[0] == 'C':
		freq = 500
	else:
		freq = 360

	#part 2 选择数据长度
	dir_pre = os.getcwd().replace('\\', '/')
	# 获得当前选择的心电图数据的路径及文件名
	ecg_file_name = dir_pre + '/mit-bih-arrhythmia-database-1.0.0/' + str(ecg_name)
	sig, fields = wfdb.rdsamp(ecg_file_name, channels=[0])
	sig_len=len(sig)

	msg="请输入数据起始点、终止点、数据通道(0,1),"
	msg+=("当前数据终止点为"+str(sig_len))
	print(msg)
	fieldNames=["*数据起始点","*数据终止点","*数据通道"]
	defaultValues=['0',str(sig_len),'0']
	fieldValues=[]
	title="ECG相关信息输入"
	fieldValues=g.multenterbox(msg,title,fieldNames,defaultValues)
	while True:
		if fieldValues==None:
			break
		errmsg=""
		for i in range(len(fieldNames)):
			option=fieldNames[i].strip()
			if fieldValues[i].strip()=="" and option[0]=='*':
				errmsg+=("【%s】为必填项" %fieldNames[i])
			elif i==2 and (int(fieldValues[i])>1 or int(fieldValues[i])<0):
				errmsg += ("【%s】只能为0或1" % fieldNames[i])
		if errmsg=="":
			break
		fieldValues = g.multenterbox(errmsg, title, fieldNames, fieldValues)
	sampfrom=int(fieldValues[0])
	sampto=int(fieldValues[1])
	channel=int(fieldValues[2])

	#part3 选择心电检测算法
	msg="请选择心电检测算法"
	title="选择心电检测算法"
	choices_list=["XQRS_Algorithm","GQRS_Algorithm","WQRS_Algorithm","SQRS_Algorithm","SQRS125_Algorithm","EcgAnalysis_Algorithm and HRV_Analysis"]
	reply=g.choicebox(msg,title,choices=choices_list)
	r_peak_inds=read_r_peak(ecg_file_name,sampfrom,sampto)
	summary=None
	if reply=="WQRS_Algorithm":
		summary=wqrs_algorithm(ecg_file_name,sampfrom,sampto,channel,r_peak_inds)
	elif reply=="XQRS_Algorithm":
		summary=xqrs_algorithm(ecg_file_name,sampfrom,sampto,channel,r_peak_inds)
	elif reply=="GQRS_Algorithm":
		summary = gqrs_algorithm(ecg_file_name, sampfrom, sampto, channel, r_peak_inds)
	elif reply=="SQRS_Algorithm":
		summary=sqrs_algorithm(ecg_file_name, sampfrom, sampto, channel, r_peak_inds,freq)
	elif reply=="SQRS125_Algorithm":
		summary=sqrs125_algorithm(ecg_file_name, sampfrom, sampto, channel, r_peak_inds,freq)
	elif reply=="EcgAnalysis_Algorithm and HRV_Analysis":
		summary=EcgAnalysis_algorithm(ecg_file_name, sampfrom, sampto, channel, r_peak_inds)
	else:
		pass

	summary_=""
	for item in summary:
		summary_+=item+'\n'
	title="结果分析"
	if reply=="EcgAnalysis_Algorithm and HRV_Analysis":
		title="结果分析以及心率变异系数"
	g.msgbox(msg=summary_, title=title)