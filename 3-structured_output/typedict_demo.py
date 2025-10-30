from typing import TypedDict

class Person(TypedDict):

    name:str
    age: int

new_person:Person = {'name':'Saiyyam','age':22}

print(new_person)