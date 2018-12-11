import os
import sys
min_year = 19
max_year = 2020
time_step = int(sys.argv[1])
directory = "./wiki_data_timestep_"+str(time_step)
HyTE = "./wiki_data"
if not os.path.exists(directory):
	os.makedirs(directory)
fp_write = open(directory+"/time2id.txt", "w")
time_window = []
ind = 0
for i in range(min_year, max_year+1, time_step):
	time_window.append((i, i+time_step))
	fp_write.write(str(i)+"\t"+str(ind)+"\n")
	ind += 1
fp_write.close()
entity2id = {}
with open(HyTE+"/entity2id.txt","r") as fp:
	for line in fp:
		line = line.strip("\n")
		line = line.split("\t")
		entity2id[line[0]] = line[1]
id2entity = {v: k for k,v in entity2id.iteritems()}

relation2id = {}
with open(HyTE+"/relation2id.txt","r") as fp:
	for line in fp:
		line = line.strip("\n")
		line = line.split("\t")
		relation2id[line[0]] = line[1]
id2relation = {v: k for k,v in relation2id.iteritems()}

time2id = {}
with open(directory+"/time2id.txt","r") as fp:
	for line in fp:
		line = line.strip("\n")
		line = line.split("\t")
		time2id[line[0]] = line[1]
id2time = {v: k for k,v in time2id.iteritems()}

fp_write = open(directory+"/train.txt", "w")
with open(HyTE+"/train.txt", "r") as fp:
	for line in fp:
		line = line.split("\t")
		left = line[3].split("-")[0]
		if left != "####":
			left = int(left)
		else:
			left = min_year
		right = line[4].split("-")[0]
		if right != "####":
			right = int(right)
		else:
			right = max_year
		for window in time_window:
			if (left < window[0] and right < window[0]) or (left > window[1] and right > window[1]):
				continue
			else:
				fp_write.write(id2entity[line[0]]+"\t"+id2entity[line[2]]+"\t"+id2relation[line[1]]+"\t"+str(window[0])+"\n")
fp_write = open(directory+"/valid.txt", "w")
with open(HyTE+"/valid.txt", "r") as fp:
	for line in fp:
		line = line.split("\t")
		left = line[3].split("-")[0]
		if left != "####":
			left = int(left)
		else:
			left = min_year
		right = line[4].split("-")[0]
		if right != "####":
			right = int(right)
		else:
			right = max_year
		for window in time_window:
			if (left < window[0] and right < window[0]) or (left > window[1] and right > window[1]):
				continue
			else:
				fp_write.write(id2entity[line[0]]+"\t"+id2entity[line[2]]+"\t"+id2relation[line[1]]+"\t"+str(window[0])+"\n")
fp_write = open(directory+"/test.txt", "w")
with open(HyTE+"/test.txt", "r") as fp:
	for line in fp:
		line = line.split("\t")
		left = line[3].split("-")[0]
		if left != "####":
			left = int(left)
		else:
			left = min_year
		right = line[4].split("-")[0]
		if right != "####":
			right = int(right)
		else:
			right = max_year
		for window in time_window:
			if (left < window[0] and right < window[0]) or (left > window[1] and right > window[1]):
				continue
			else:
				fp_write.write(id2entity[line[0]]+"\t"+id2entity[line[2]]+"\t"+id2relation[line[1]]+"\t"+str(window[0])+"\n")
