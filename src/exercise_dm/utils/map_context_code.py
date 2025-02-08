import csv
import json
# Load the data
code_context_map_v1 = {}
code_context_map_v2 = {}

with open('data/codes.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(csvreader)  # Skip the header (id,main,target,context,score)

    context = ""
    last_code = ""
    for row in csvreader:
        code = row[0]
        if len(code) == 3:
            code_context_map_v1[code] = row[1]
            if last_code:
                code_context_map_v2[last_code] = context
            last_code = code
            context += row[1]
        if len(code) == 4:
            context += f'; {row[1]}'
    code_context_map_v2[last_code] = context      

json.dump(code_context_map_v1, open('data/code_context_map_v1.json', 'w'))
json.dump(code_context_map_v2, open('data/code_context_map_v2.json', 'w'))