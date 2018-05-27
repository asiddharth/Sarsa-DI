from os import listdir
from os.path import isfile, join
import math
x1 =[0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.999, 0.9999, 0.99999, 1]

for i in x1 :
	onlyfiles = [f for f in listdir(".") if isfile(join(".", f))]
	files = [x for x in onlyfiles if x.startswith("lambda1_1_" + str(i) + "_")]
	mean_dict = {}
	stderr_dict = {}
	for datafile in files :
		f = open(datafile, 'r')
		content = f.read().splitlines()
		for line in content[2:-2] :
			data = line.split("\t")
			mean_dict[int(data[0])] = mean_dict.get(int(data[0]),0) + float(data[1])
		f.close()
	for key, value in mean_dict.items():
		mean_dict[key] = value / len(files)
	
	for datafile in files :
		f = open(datafile, 'r')
		content = f.read().splitlines()
		for line in content[2:-2] :
			data = line.split("\t")
			stderr_dict[int(data[0])] = stderr_dict.get(int(data[0]),0)  + (float(data[1]) - mean_dict[int(data[0])]) ** 2
		f.close()
	for key, value in stderr_dict.items():
		stderr_dict[key] = math.sqrt(value) / len(files)
	print len(files)
	keylist = mean_dict.keys()
	keylist.sort()
        f = open("data" + str(i) + "avg", 'w')
	for key in keylist:
		if key <= 500000 : 
			f.write (str(key) + " " + str(mean_dict[key]) + " " +str( (stderr_dict[key]))+ "\n")
			#f.write (str(key) + " " + str(mean_dict[key]) + " " +str(mean_dict[key] - (stderr_dict[key])/2)+ " " +str(mean_dict[key] - (stderr_dict[key])/2)+ " " +str(mean_dict[key] + (stderr_dict[key])/2)+ " " +str(mean_dict[key] + (stderr_dict[key])/2)+ "\n")
	f.close()
