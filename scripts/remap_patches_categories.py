# Python program to read
# json file

import json
import sys

def replace_category_of_path(file_path):

    print(file_path)
    # Read Content in patch file
    with open(file_path, 'r') as file:
        content = file.read()
        file.close()

    # Write replaced content in patch files
    with open(file_path, 'w') as file:
        for categ in category["categories"]:
            content = content.replace(categ["name"], categ["supercategory"])
        file.write(content)
        file.close()


if len(sys.argv) < 2:
    print("\nThis script gets patch folder as parameter\n")
    sys.exit(0)

categorypath = "..\jsonfiles\categories.json"

# Get category data
category_json = open(categorypath)
category = json.load(category_json)
category_json.close()

labelbasepath = sys.argv[1]
# Opening JSON file

label_paths = ["real\\test","real\\training","syntehtic\\test","syntehtic\\training","syntehtic\\validation"]

for label_path in label_paths:
    replace_category_of_path(labelbasepath + label_path + "\labels.json")

