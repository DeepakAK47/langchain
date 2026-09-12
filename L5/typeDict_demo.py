# Use of Type Dict : as providing the structured data
from typing import TypedDict

class Person(TypedDict):
    name :str
    age : int
    email : str

new_person : Person = {
    'name' : "Deepak Singh",
    'age' : 25,
    'email' : "Deepak@gmail.com"
}

print(new_person['name'])