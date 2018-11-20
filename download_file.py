# import urllib.request
from tqdm import tqdm

# a = [str(i) for i in range(100,234)]
# a=[107,108]
# for i in tqdm(a):
# 	try:
# 		url = "https://physionet.org/atm/mitdb/"+i+"/atr/0/e/rdann/x/annotations.txt"
# 		urlm = "https://physionet.org/atm/mitdb/"+i+"/atr/0/e/export/matlab/"+i+"m.mat"

# 		file_name = url.split('/')[-7]+'_'+url.split('/')[-1]
# 		file_namem = url.split('/')[-1]

# 		f = open(file_name, 'wb')
# 		urllib.request.urlretrieve(url, './data/'+file_name)
# 		f.close()
		
# 		f = open(file_namem, 'wb')
# 		urllib.request.urlretrieve(urlm, './data/'+file_namem)
# 		f.close()
# 	except:
# 		pass

# f = open('./data/download.txt','w')
import os
a = [str(i) for i in range(100,234)]
for i in tqdm(a[:7]):
	# os.system("wget -P ./data \"https://physionet.org/atm/mitdb/"+i+"/atr/0/e/export/matlab/"+i+"m.mat\""+"\n")
	os.system("wget -O ./atr_data/"+i+"_atr.txt \"https://physionet.org/atm/mitdb/"+i+"/atr/0/e/rdann/x/annotations.txt\""+"\n")
for i in tqdm(a[7:]):
	# os.system("wget -P ./data \"https://physionet.org/atm/mitdb/"+i+"/atr/0/e/export/matlab/"+i+"m.mat\""+"\n")
	os.system("wget -O ./atr_data/"+i+"_atr.txt \"https://physionet.org/atm/mitdb/"+i+"/atr/0/e/rdann/annotations.txt\""+"\n")

