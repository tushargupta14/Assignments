## cONFUSION MATRIX 
## false negatives and tagged now both present 
import json 
import os
from collections import defaultdict
def process_ann_file(matrix_dict,file_path) :

	with open(file_path,"rb+") as f :
		for line in f :
			elements = line.rstrip().split() 
			if "#" in elements[0] :
				continue
			entity = elements[1]
			if matrix_dict[entity]['true_positives'] is None :
				matrix_dict[entity]['true_positives'] = 1
			else :
				matrix_dict[entity]['true_positives']+=1
			

def read_entity_tags(matrix_dict) :


	with open("entity_tags.txt","rb+") as f :

		for line in f :

			elements = line.rstrip().split("\t",2)
			typ = elements[1]
			info =  elements[2]
			
			info = info.lower()
			#print typ,info
			desc = ""
			if "tagged" in info or "negative" in info:
				if matrix_dict[typ]['false_negatives'] is None :
					matrix_dict[typ]['false_negatives'] = 1
				else :
					matrix_dict[typ]['false_negatives']+=1

			if "positive" in info or "to" in info :
				if matrix_dict[typ]['false_positives'] is None :
					matrix_dict[typ]['false_positives'] = 1
				else :
					matrix_dict[typ]['false_positives']+=1
			if "sentiment" in info :
				info = info.replace(",","")
				info = info.replace("'","")
				info = info.replace(" ","")
				print info
				
				if 
				
			#if "sentiment:high" in info:
				#if info.index("to") > info.index("high") :
					#matrix_dict["Sentiment"]['false_positives']+=1
				#else :
					#matrix_dict["Sentiment"]['false_negatives']+=1
				#print info
			#if "sentiment" in info and "to" in info :
				#print "ded",info 
	return matrix_dict
def read_all_files() : 
	
	count = 0 
	total = 0 

	matrix_dict = defaultdict(lambda: dict.fromkeys(["false_positives","false_negatives","true_positives"]))

	for dirs,subdirs,files in os.walk("./edge/") :

		for f in files :
			if f.endswith(".ann") :
				total+=1
				file_path =  os.path.join(dirs,f)
				#print file_path
				process_ann_file(matrix_dict,file_path)	
	return matrix_dict
if __name__ == "__main__" :


	matrix_dict = read_all_files()
	matrix_dict = read_entity_tags(matrix_dict)

	#print matrix_dict

	with open("metrics1.json","wb+") as f :
		json.dump(matrix_dict,f)

