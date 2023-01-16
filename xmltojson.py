import json
import os

from bs4 import BeautifulSoup

for filename in os.listdir("data/Annotations"):
    
    #Load xml
    xml_parser = BeautifulSoup(open('data.xml'), 'xml')

#Extract relevant information
name = xml_parser.find('name').contents[0]
age = xml_parser.find('age').contents[0]
role = xml_parser.find('role').contents[0]

employee = {
    'name':name,
    'age': age,
    'role': role
}
print(json.dumps(employee))