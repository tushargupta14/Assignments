import os 
import json

from collections import defaultdict

def process_file(output_file,file_path) :

	file_dict = {}
	with open(file_path,"rb+") as f :

		for line in f :
			desc = ""
			elements = line.rstrip().split() 
			if "#" in elements[0] :
				qid = elements[2]
				description = elements[3:]
				file_dict[qid] = description
				for el in description :
					desc+=el

				if "Sentiment" in desc :
					print file_path
					print elements
					print desc
	if len(file_dict) ==0 :
		return 
	with open(file_path,"rb+") as f:
		for line in f :
			elements = line.rstrip().split()

			if elements[0] in file_dict :
				output_file.write("{}\t{}\t{}\n".format(elements[0],elements[1],file_dict[elements[0]]))

def read_ann_file() :

	output_file = open("entity_tags_2.txt","wb+")
	count = 0 
	total = 0 
	for dirs,subdirs,files in os.walk("./edge/") :

		for f in files :
			if f.endswith(".ann") :
				total+=1
				file_path =  os.path.join(dirs,f)
				#print file_path
				process_file(output_file,file_path)

def read_output_file() :

	stat_dict = defaultdict(lambda : defaultdict(int))
	with open("entity_tags.txt","rb+") as f :

		for line in f :

			elements = line.rstrip().split()
			typ = elements[1]
			info = elements[2]+elements[3]

			info = info.lower()

			if "tagged" in info :
				stat_dict[typ]["tagged_now"]+=1
			if "negative" in info :
				stat_dict[typ]["false_negative"]+=1
			if "positive" in info :
				stat_dict[typ]["false_positive"]+=1

	#print stat_dict
	with open("stat_dict.json","wb+") as f :
		json.dump(stat_dict,f)



if __name__ == "__main__" :

	read_ann_file()
	read_output_file()