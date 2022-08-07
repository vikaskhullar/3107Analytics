# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 19:32:07 2022

@author: vikas
"""

c = 'A'
ord(c)

c = 'Z'
ord(c)

c = ';'
ord(c)

c = ' '
ord(c)


a = 65
chr(a)


s = "Welcome to python"
for i in s:
    if(ord(i)==32):
        print("space idenfied")


# Password should be between A-Z single char

password = ';'

#Dont focus on logic till time
if (ord(password)>=65 and ord(password)<=90):
    print("Correct")
else:
    print("Not Correct")


import string
string.punctuation




#Numeric
x = 5

#Operators - Airthmetic


print(x)

print(type(x))

print(x -1)
print(x + 1)
print(x * 2)
print(x / 2)

print( x ** 2)
print( x ** 3)
print( x ** 4)

print( x ** (1/2))
print( x ** (1/3))


#Modulus
print(5 % 2)

a = 12

'''
if (a%2 == 0):
    print("Even")
else:
    print("Odd")
'''

y = x * 2       
x
y

y = x + 2
y
   

# Boolean Datatype

t = True
f = False    

a = True
b = False    

'''
AND
A B O
F F F
F T F
T F F
T T T

OR
A B O
F F F
F T T
T F T
T T T
'''

a=10
b=20
c=15

a<b
a>b
b<c

if (a<b and b<c):
    print("C is Greatest")


print(f and f)
print(t and f)
print(f and t)
print(t and t)

print(f or f)
print(t or f)
print(f or t)
print(t or t)


# Conditional Operators

a = 10

b = a # assignment operator

a = 10
b = 20

a == b # conditional equals to check

a < b
a > b

a<=b
a>=b

a != b

#String Handling Operations

h = 'hello'
w = 'world'

hw = h+' ' + w

hw

hw = hw.upper()

hw = hw.lower()

hw =hw.capitalize()


s = "Welcome to Java"
s = s.replace("Java", "Python")


'''
r = hw.split(' ')
s=''
for i in r:
   s = s+ i.capitalize() + ' ' 
s
'''

name = "Ajay-Kumar"
name = name.split('-')
name 

Fname = name[0]
Lname =name[1]

Fname
Lname

jname = Fname + ' ' + Lname
jname


'''
s = "Welcome to Python"
s =s.split(' ')
s.reverse()
a =''
for i in s:
    a=a+i+' '
a
'''


name = '     Jay Kumar      '
name  = name.strip()

name = 'Jay Kumar Mishra'
name = name.replace(' ','')
name




'''
def replacefirst(name):
    n=''
    flg=0
    for i in name:
        if(i !=' '):
            n=n+i
        elif(i==' '):
            if(flg==0):
                flg=1            
            else:
                n=n+i
    return(n)

name = 'Jay Kumar Mishra Kumar'
replacefirst(name)
'''
'''
name = 'Jay Kumar Mishra Kumar'

name.split()

flg=0
r=''
for i in name.split():
    if (flg==1):
        r =r+' '+i
    else:
        r=r
        flg=0
    r=r+i
r
'''


# integer float string boolean
# Operators Airthmatic and conditional

a = 10 

# List, tuple, Set, Dictionary
# Properties, changable or not changable, indexed or not indexed,
#ordered or not ordered, hetro or homo

# List

l1 = []

l2 = [10, 20, 30, 40, 50]

l2
print(l2)

#Individual Values
#Indexed

l2[0]
l2[1]
l2[2]
l2[3]
l2[4]
l2[5]

# Hetrogenoeous 
#Different Type

l3 = [10, 34.5, True, 'SK']


# Mutable or Changable
l3[0] = 20
l3[3] = 10.02


# Not Odered
l4 = [30, 10, 40, 20]
l4
l4[0]
l4[1]

l4

l4[0]
l4[1]
l4[2]
l4[3]



r1 = range(100)
r1
l5 = list(r1)

l5[0]
l5[1]


r2 = range(100, 10000)
l6 = list(r2)
l6



for i in l6:
    print(i)

l4
for i in l4:
    print(i*2)

r3 = list(range(10))
r3

r4 = list(range(5,21))
r4

r5 = list(range(5,51, 5))
r5

#Function related to List
l7 = [30, 60, 20, 10]
l7

#Never declare a variable which starts with numeric

l7.append(50)

l7.append('VK')

print(l7.pop())

print(l7.pop())

l7.append(70)

# Indexed based deleting
l7.pop(2)

l7.insert(2, 66)

l7
l7.sort()
l7

l7[0]
l7[4]

l7 = [30, 60, 20, 10]
l7.sort()
l7.reverse()
l7[0]
l7[4]


# tuple
t1 = ()

#Hetrogeneous
t2 = (20,4.6, True, 'SK')
t2

#Indexed
t2[0]
t2[1]
t2[2]
t2[3]
t2[4]

#No Ordered

# Not mutable or not changable
t2[0] = 10 
#TypeError: 'tuple' object does not support item assignment
t2.append(100)
#AttributeError: 'tuple' object has no attribute 'append'
t2.pop()
#AttributeError: 'tuple' object has no attribute 'pop'

for i in t2:
    print(i)


'''
l6 = [10, '20', 30]
def num(l6):
    l7=[]
    for i in l6:
        if(type(i)==int):
            l7.append(i)
    return(l7)
            
num(l6)
'''

#Sets
s1 = {10}

#Ordered
s2 = {40, 20, 30, 10}
s2

#No Indexed
s2[0]
#TypeError: 'set' object is not subscriptable

for i in s2:
    print(i)

#Mutable or Changable
s2.add(15)
s2

#holding unique values
s2 = {20, 40, 30, 20,50, 20}
s2

s2.update([55, 33])
s2

s2.remove(33)
s2.remove(33)
#KeyError: 33

s2.discard(10)
s2.discard(10)

s2.pop()
s2
s2.pop()

#Set Operations
s1 = {10, 20, 30, 40}
s2 = {30, 40, 50, 60}

#Union
s3 = s1.union(s2)
s3

#Intersection
s4 = s1.intersection(s2)
s4

#Differece
s5 = s1.difference(s2)
s5


#Dictionary

d1 = {}

#not indexed
#Key value paired

cars = {'brand':'Honda', 'year': 2022, 'color':'black'}
cars

#access data through keys
cars['brand']
cars['year']

# allow to add or delete it is mutable or changable
cars['model'] = 'Jazz'
cars

#functions
cars.keys()
cars.values()
cars.items()

c = [('brand', 'Honda'), ('year', 2022), ('color', 'black'), ('model', 'Jazz')]

for i in cars.keys():
    print(cars[i])

for i in cars.values():
    print(i)

for i in cars.items():
    print(i)

for i,j in cars.items():
    print(i)
    print(j)
    
    
#Operations

cars.pop('model')
cars
cars.popitem()
cars.popitem()
cars['brand'] = 'Tata'






































































 







    
    
    






'''


'''









































































































