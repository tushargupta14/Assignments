## Assumption : Text files contain random texts. The program only prints the telephone numbers present in the following formats :
## +91-9547010744 ; +919547010744 ; 9547010744 ; 09547010744 ;

import os 
import sys 
import re 


# Obtaining a list of all text files in the directory 

# Insert path to directory containing text files
path_to_directory  = "/target_directory/../"

for dirs,subdirs,files in os.walk(path_to_directory):
	for filename in files:
		if filename.endswith(".txt"):
			#print filename
			with open(os.path.join(dirs,filename),"rb+") as f:
				for line in f:
					matches = re.findall(r'((\+91[\-\s]?)*[789]\d{9}$\b)|((0)?[789]\d{9}\b)',line.rstrip("\n"))
					#print matches
					for match in matches:
						for match_groups in match :
							if len(match_groups) > 8:
								print match_groups
