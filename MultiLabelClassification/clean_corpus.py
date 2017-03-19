# Get the unique no. of job functions 
# Removes the stop words, punctuations and numerals from the corpus

import re
import csv 
import json 
def save_labels():


	job_functions = set()
	
	input_file  = open("data/dataset.csv","rb+")

	reader = csv.reader(input_file)

	for row in reader :

		function = row[0]
		for func in function.split(",") :
			job_functions.add(func)

	job_functions.remove("function")

	print len(job_functions)
	function_list = list(job_functions)
	print len(function_list)

	function_label_dict = {}

	for i in range(len(function_list)) :
		function_label_dict[function_list[i]] = i

	with open("data/function_labels.json","wb+") as f :
		json.dump(function_label_dict,f)
def pre_process(parag) :
	
	# Removing stop words, punctuations and numerals
	clean_parag = re.sub("[^a-zA-Z]"," ",parag)
	#clean_parag  = re.sub(" +"," ",clean_parag)
	clean_parag = re.sub("\n","",clean_parag)
	clean_parag = clean_parag.lower()
	stop_set = []
	clean_parag = " ".join([w for w in clean_parag.split() if len(w) >1 ])

	with open("data/stop_words.txt","rb+") as f :
		stop_set = [re.sub("\n","",line) for line in f]

	word_set = [word for word in clean_parag.split(" ")]

	intersection = [word for word in word_set if word not in stop_set]
	clean_parag = " ".join(w for w in intersection)
	return clean_parag


def clean_corpus() :

# Cleans the corpus of stop words and removes NA
	output_file = open("data/dataset.csv","a")
	count =0 
	job_functions = set()

	writer = csv.writer(output_file,delimiter = ",")
	with open("data/function_assignment.csv") as f :

		reader = csv.reader(f)
		for row in reader :
			#print row 
			description = row[2]
			if 'NA' == description :
				#print description
				continue
			function = row[0]
			for func in function.split(",") :
				job_functions.add(func)
			title = row[1]
			
			description = pre_process(description)
			writer.writerow([function,title,description])

			count+=1 
		print len(job_functions)
		print count


	
if __name__ == "__main__" :

	
	clean_corpus()
	save_labels()