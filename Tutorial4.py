# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:04:23 2019

@author: Sam Kettlewell
"""

#Benchmarking
import time
import timeit

def procedure():
    time.sleep(2.5) #measured in seconds
    return None

t_start = time.time()
#time.time() yields number of seconds since origin (Jan 1st 1970)
#time.clock() behaves like a stopwatch.
#God knows what perf counter does
#time.process_time() does not include time spent sleeping
    
#t0 = time.perf_counter()
#procedure()
#t1 = time.perf_counter()

#print(t1-t0)
#print("")

t1 = timeit.timeit(stmt="lst = ['hello'] * 100", number = 100) #Note this changes the order of magnitude
t2 = timeit.timeit(stmt="lst = ['hello' for i in range(100)]")

t_end = time.time()

print(t1)
print(t2)
print(t_end - t_start)

#This runs one million trials by default

#We can also run functions with the timeit module. See the code in tutorials.

#Now to compute determinants
