import matplotlib.pyplot as plt
import glob
import math
keys = {}

for dir in glob.glob("data/*"):
	try:
		file1  = open(dir + "/VBC")
		file2  = open(dir + "/VAB")
		file3  = open(dir + "/VCA")
	except:
		print("Dir " + dir + " broked")
		continue
	if dir == "data/KELLER_HALL_MAIN_MTR":
		continue
	data_min = []
	data_max = []
	t_min = []
	t_max = []
	print(dir)
	for line1 in file1:
		line2 = file2.readline();
		line3 = file3.readline();
		min1 = float(line1.split()[2])
		min2 = float(line2.split()[2])
		min3 = float(line3.split()[2])
		min_rms = math.sqrt(min1**2 + min2**2 + min3**2)/4/math.sqrt(3)
		if min_rms > 60:
			t_min.append(int(line1.split()[0]))
			data_min.append(min_rms)
		
		max1 = float(line1.split()[3])
		max2 = float(line2.split()[3])
		max3 = float(line3.split()[3])
		max_rms = math.sqrt(max1**2 + max2**2 + max3**2)/4/math.sqrt(3)
		if max_rms > 60:
			t_max.append(int(line1.split()[0]))
			data_max.append(max_rms)
		if data_max[-1] > 125 or data_min[-1] <115:
			if not line1.split()[0] in keys:
				keys[line1.split()[0]] = []
			keys[line1.split()[0]].append(dir.split("/")[1])		
	plt.xlim((1574889300 - 10*60, 1574889300 + 10*60))
	plt.title("Device: " + line1.split()[0])
	plt.plot(t_min, data_min)
	plt.plot(t_max, data_max)
	plt.show()
	#break;

for key in keys:
	if len(keys[key]) >1:
		print(key)
		for meter in keys[key]:
			print("\t" + meter)
