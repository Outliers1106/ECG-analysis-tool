# ECG-analysis-tool
ECG分析可视化工具，使用简单的GUI展示
## 安装过程
- 首先在 https://github.com/Outliers1106/ECG-analysis-tool 进行下载，解压
- 安装Anaconda3
   - 在 https://www.anaconda.com/distribution/#download-section 下载Anaconda3，并进行安装
   - 安装完成后，打开Anaconda Prompt，输入`conda create -n your_env_name python=3.7`，创建虚拟环境
   - 创建虚拟环境之后，cd到项目目录 XXX/ECG-analysis-tool，输入`activate your_env_name`，启动虚拟环境
   - 下载项目运行所需的包`pip install easygui`、`pip install wfdb`、`pip install matplotlib`，为了加快下载速度，可以使用清华源，在指令后加上`-i https://pypi.tuna.tsinghua.edu.cn/simple` 
- 运行项目
   - 打开Anaconda Prompt，进入项目目录，启动虚拟环境
   - 输入`python easy_gui.py`即可运行程序**如果提示 WinError 126，那么把ECG-analysis-tool/DLL下的动态链接库文件全部放到ECG-analysis-tool目录下，再次运行程序即可**
   
Project_qrs.dll 来自repo [link](https://github.com/Outliers1106/physionet_qrs_algorithm)

## 运行结果展示
#### 一、测试单条数据
- 1.选择进行测试的数据集（MIT/CPSC）

- 2.选择数据（单选）

![image](https://github.com/Outliers1106/ECG-analysis-tool/blob/master/img/pic1.png = 200x200)

- 3.选择数据起始点和数据通道（默认为0）

![image](https://github.com/Outliers1106/ECG-analysis-tool/blob/master/img/pic2.png)

- 4.选择算法（可多选）

![image](https://github.com/Outliers1106/ECG-analysis-tool/blob/master/img/pic3.png)

- 5.结果展示与分析

![image](https://github.com/Outliers1106/ECG-analysis-tool/blob/master/img/pic4.png)
![image](https://github.com/Outliers1106/ECG-analysis-tool/blob/master/img/pic5.png)

#### 二、测试多条数据
- 1.选择进行测试的数据集（MIT/CPSC）

- 2.如果选择MIT数据集，直接选择多行数据即可，如果选择CPSC数据集，即可直接选择多行数据，也可选中第一行"MultipleSelection"，然后手动输入选择数据的范围

- 3.选择算法，单选

- 4.程序自动输出该算法在所选数据的测试统计结果，不进行画图
