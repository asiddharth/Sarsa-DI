from os import listdir
from os.path import isfile, join
import math
onlyfiles = [f for f in listdir(".") if isfile(join(".", f))]
files = [x for x in onlyfiles if x.startswith("data") and x.endswith("avg")]
for datafile in files :
	f=open(datafile, 'r') 
	content = f.read().splitlines()
	maximum=-float("inf")
	var = 0
	for line in content[-6:] :
		val = float(line.split(" ")[1])
		if val > maximum :
			 maximum = val
			 var = float(line.split(" ")[2])
	if datafile[4:-3] == "0.99999":
		continue
	print datafile[4:-3], " ", maximum, var
