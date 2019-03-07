# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:01:08 2019

@author: Sam Kettlewell
"""

""" FILE STUFF
file = open('datafile.txt', 'w') #Codes 'w' is write; 'r' is read; 'a' is 'append to'

try:
    print(file.read())
except:
    print("not readable")


try:
    for x in range(10):
        file.write('stuff')
except:
    print("not writable")

file.close()

#Opening the same file again overwrites the contents of the file.
#with syntax also works here.
"""

import numpy as np

#Vector
v1 = np.array([1,2,3])
print(v1)

#matrix
m1 = np.array([[1,2], [3,4]])
print(m1, m1[0,1]) #Zeroth row, first column.

print(m1.T) #Transpose of m1
print(m1.shape) #Dimensions of m1
print(m1.size)

#To compare lists and numpy arrays:
list1 = [1,4,9]
list2 = [2,5,10]

print(list1 + list2) #Concatenates them
print(np.array(list1) + np.array(list2)) #Adds/multiplies/etc... componentwise

m2 = np.array([[4,9,5], [3,2,7]])
m3 = np.array([[1,2], [3,4]]) #Two matrices which 'technically' can be multiplied

#print(m1*m2) - Throws error. Instead
print(m3.dot(m2)) #Matrix mutiplication. ENSURE CORRECT WAY ROUND - NOT COMMUTATIVE
print(m3@m2) #Also matrix multiplication - shorthand

#We can create an empty matrix (4x4) by
empty = np.zeros([4,4])
print(empty)
#We can also use np.ones

#np.arange() is the same as vanilla Python's range() function but takes non-integer arguments
range_array = np.arange(0.5, 0.9, 0.05)
print(range_array)

#np.linspace gives us the correct scaling for an interval given its start, end and no. of points needed
print(np.linspace(1.0, 2.5, 8)) #THIS ONE IS APPARENTLY USEFUL

#We can write numpy objects directly to files by using
#np.savetxt('numpyfile.txt', m1) - Save the matrix to the file.

#We can read back using np.loadtxt(), this ensures numpy recognises it as an object from the numpy module.
print(np.random.random(12)) #Array of length 10 i.i.d Unif(0,1) r.v
print(np.random.normal(0,1,10)) #Array of length 10 i.i.d N(0,1) r.v