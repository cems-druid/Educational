#Serialization: Transforming data into a series of bytes to be transmitted across a network.
#Deserialization: Decoding data that has been stored or delivered in the JSON standard.

import json

data = {
    "president" : {
        "name" : "Ahmet Ahmet",
        "species" : "Kasim"
        
    }
}

#cretes a file called "data1.json", writes into the file.
with open("data1.json", "w") as write_file:
    json.dump(data, write_file)


#very suprisingly, changes dict (json) into string
json_string = json.dumps(data)
json_string1 = json.dumps(data, indent=2)
print(type(json_string), '\n', type(data))
print(json_string1, '\n', data)
