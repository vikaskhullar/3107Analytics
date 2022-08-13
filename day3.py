# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 19:41:40 2022

@author: vikas
"""

s = {4,2,3,5,5,10,21,12}
s1 = {4,2,3,5,5,10,21,12}

s.union(s1)


l1 = [3,2,4]
l1.pop()
s.pop()

d = {'a':1, 'b':2, 'c':3}
d.popitem()


#control Statements 

#Conditional Control Statements
#Looping Control statements



#Conditional Control Statements
'''
if condition true:
    allow
else:
    not allow

if condition:
    statement1
    statement2
else:
    statement3  



if (a<10)
{
    sub statements 
}

'''

a = 10
a>0 # conditional check statements always return true or false

if (a>0):
    print("A is greater than 0") #Intendation
    print("a")
print("Outside")


a=-1
a>0

if (a>0):
    print("A is greater than 0") #Intendation
    print("a")
print("Outside")


age=19

if (age>=18):
    print("Right to Vote")


age=16
if (age>=18):
    print("Right to Vote")
else:
    print("No voting")


a = 10
b = 10

if (a>b):
    print("a is greater")
else:
    print("b is greater")


a = 10
b = 10

if (a>b):
    print("a is greater")
elif (b>a):
    print("b is greater")
else: # default call
    print("A and b are equal")


'''
Marks           Grade
<=50             F
>50 and <=60     E
>60 and <=70     D
<70 and <=80     C
<80 and <=90     B
<90              A
'''


marks = 85
marks>90

marks>80 and marks<=90



marks=45
if (marks>90):
    print('A')
elif(marks>80 and marks<=90):
    print('B')
elif(marks>70 and marks<=80):
    print('C')
elif(marks>60 and marks<=70):
    print('D')
elif(marks>50 and marks<=60):
    print('E')
else:
    print('F')



v = 2222222222

if (v%2==0):
    print("Even Number")
else:
    print("Odd Number")




# Looping or Iterative Statements


teamA = ['India', 'Australia','Pakistan', 'England']   # 4elements   list index 0-3


print(teamA[0])
print(teamA[1])
print(teamA[2])
print(teamA[3])
print(teamA[4]) #IndexError: list index out of range


for i in teamA:
    print(i)


r1 = range(1,221)

list(r1)


for i in range(1,11):
    print(i)
    

'''
2*1=2
2*2=4
2*3=6
2*4=8
2*5=10
2*6=12
2*7=14
2*8=16
2*9=18
2*10=20
'''


lc = 'India'
wc ='USA'

print( "I live in lc and I work in wc")
print( "I live in {lc} and I work in {wc}")
print( f"I live in {lc} and I work in {wc}")


for i in range(1,11):
    print(f"2*{i}={2*i}")


print("Start Table")
for i in range(1,11):
    print(f"2*{i}={2*i}")
print("End Table")


# Nested Loops

for j in range(2,5):
    print(f"Table {j} Started")
    for i in range(1,11):
        print(f"{j}*{i}={j*i}")
    print(f"Table {j} Ended")
print("All Tables Completed")



# While Loops

cnt = 1
while(cnt<=10):
    print(cnt)
    cnt = cnt + 1



cnt = 1
while(cnt<=10):
    print(cnt)
#Operate Infinitely



j = 2
i = 1
while(i<=10):
    print(f"{j}*{i}={j*i}")
    i = i+1


j = 2
while(j<=4):    
    i = 1
    while(i<=10):
        print(f"{j}*{i}={j*i}")
        i = i+1
    j=j+1



#break and continue

teamA = ['India', 'Australia','Nepal', 'England']

for i in teamA:
    print(i)



for i in teamA:
    print(i)
    if (i=='Nepal'):
        print("Found")
        break


import numpy as np
lst = list(np.random.randint(1, 100, size=10))



lst


for i in lst:
    print(i)
    if(i==57):
        print("Element found")
        break
        


for i in lst:
    print(i)
    break
        

57 in lst




#Continue


for i in teamA:
    if (i=='Nepal'):
        print("Found")
        break
    print(i)



for i in teamA:
    if (i=='Nepal'):
        print("Found")
    print(i)



for i in teamA:
    if (i=='Nepal'):
        print("Found")
        continue
    print(i)



name  = 'Amit Kumar Singh Gehlawat'
fn = ''
flg=0

for i in name:
    if(i==' ' and flg==0):
        flg=1
        continue
    fn=fn+i
    print(fn)
        
fn

for i in name:
    print(i)




#Functions

# System Defined Function

print('hi') 
sum((2,3))


# User Defined Functions


def cal():
    a=10
    b=20
    print(a+b)
    print(a-b)
    print(a*b)
    print(a/b)


cal()
cal()
cal()
cal()


def cal1(a,b):
    print(a+b)
    print(a-b)
    print(a*b)
    print(a/b)

cal1(10,20)

cal1(40,30)

cal1(50,20)


def printhello(name):
    print(f"Hello {name}")


printhello('Amit')

printhello('Anjali')



def evenodd(num):
    if(num%2==0):
        print("Even")
    else:
        print('Odd')

a = evenodd(10)
print(a)


evenodd(221)


lst = [4,3,17,6,4]

a  = max(lst)
print(a)

min(lst)


def maxim(ele):
    m=0
    for e in ele:
        if (m<e):
            m=e
    return(m)

lst = [4,3,17,6,4]
a = maxim(lst)

print(a)


def emp(empid, empname, empmobile, empemail=None):
    print(empid, empname, empmobile, empemail)


emp(1, 'V', 99, 'sks@gmail.com')

emp(1, 'V', 99)


#Numpy

import numpy

numpy.__version__

import numpy as np

np.__version__


r1 = np.arange(11)
r1

r2 = np.arange(1,11)
r2

r3 = np.arange(0,101, 5)
r3


#Random Generators

n = np.random.randint(1,10)
n

n1 = np.random.randint(1000,100000, size=5000)
n1

n1 = np.random.randint(1,10, size=5)
n1
n1.shape

n2 = np.random.randint(1, 10, size = (3,4))
n2
n2.shape

n3 = np.random.randint(1,10, size=(2, 3,5))
n3
n3.shape


#Indexing


n = np.random.randint(1,20, size=10)
n
n.shape

n[0]
n[0:4]
n[:4]
n[4:]
n[4:7]
n[9]
n[-1]
n[-2]
n[-3]
n[:-3]
n[-5:-2]




n = np.random.randint(1,20, size=10)
n
n[-3:]


n = np.random.randint(1,10, size=(5,4))
n
n.shape
n[0][0]
n[0,0]

n[0,:]
n[1,:]
n[2,:]
n[3,:]
n[4,:]

n

n[0:2,:]
n[-1,:]
n[-2,:]
n[-3:,:]

n

n[:,0]

n[:,1]

n[:,2]

n[:,3]

n[:,-1]
n

n[:,-3:]

n
n[1,0]
n[3,2]
n[2,1:3]

n[1:2,2:3]

n[1:3,2:]
n


n = np.random.randint(1, 10, size=(3,5,6))
n

n[0]
n[1]
n[2]


n = np.random.randint(1, 10, size=(2,3,4))
n

n[0,0,:]

n[1,1,0]



















#Array is homogeneous

#Arrays are Indexed





























































































    
    
    

































































