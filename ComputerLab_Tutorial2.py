# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:07:45 2019

@author: Sam Kettlewell
"""
#Import relevent methods from math module.
from matplotlib import pyplot as plt
from math import log, ceil, tan, atan, exp

#This is any arbitrary g(x), g'(x) for testing the functions. It can be deleted afterwards.
def g(x):
    return 3*x**2 + 12*x - 46

def g_prime(x):
    return 6*x + 12

"""Given a function f and three numbers: left, right, tol>0 (default 10^-6). Bisect
the interval [left, right] until a root of f is found to given tol."""
def bisect(f, left, right, tol = 10**-6):    
    #For interval bisection, the number of iterations required for a given tolerence is
    max_iterations = ceil((log(right-left) - log(tol))/log(2))
    
    #Iterate through as many as required to reach tolerence    
    for iterations in range(max_iterations):

        #midpoint of the interval
        midpoint = (left + right) / 2
        
        #Function values at endpoints and midpoint of interval
        f_left = f(left)
        f_right = f(right)    
        f_mid = f(midpoint)
        
        #If there is a sign change inbetween left and mid, the root lies between them
        #so bring the right hand side to mid to halve the interval.
        if f_left*f_mid <= 0:
            right = midpoint
            
        #If there is a sign change inbetween right and mid, the root lies between them
        #so bring the left hand side to mid to halve the interval.
        elif f_right*f_mid <= 0:
            left = midpoint
            
    #Return the value of the midpoint. The number of iterations required to reach tol is precisely max_iterations
    return (midpoint, max_iterations)
    
"""Use lambda functions to solve the following to reasonable tolerence"""
#Say for example we didn't know where the roots were located. How do we choose
#a sensible interval/know how many intervals we need to inspect in that case?
print("The roots of x^3 + 2x^2 - x - 1 are: ")
print(bisect(lambda x: x**3 + 2*x**2 - x - 1, -3, -2, 0.001))
print(bisect(lambda x: x**3 + 2*x**2 - x - 1, -1, 0, 0.001))
print(bisect(lambda x: x**3 + 2*x**2 - x - 1, 0, 1, 0.001))
print("") #newline

#To find the cube root of 7 we must find the root of f(x)=x^3-7. It's obviously
#positive and obviosuly less than 7 so
print("The cube root of 7 is ")
print(bisect(lambda x: x**3 - 7, 0, 7, 0.001))
print("") #newline

#Firstly rearrange to get the equation for x: x=tan(3-x^2). Plot it to find sensible
#intervals. Then,

"""These 2 are wrong. The root finding method is working fine but somehow, rearranging
the equations is producing the wrong roots. What is going on here? Am I being daft?
Please advise. (NB: I can't even check the answers using the other methods because it's
an issue with the equations not the algorithms themselves. I think)"""

print("The equation arctan(x) = 3-x^2 has solutions: ")
print(bisect(lambda x: tan(3-x**2), -2.5, -2, 0.0000001))
print(bisect(lambda x: tan(3-x**2), 1, 2, 0.0000001)) #We shouldh have x=-2.028, x=1,428
print("") #newline #It's finding roots, the equation is wrong

print("The equation log(x^4) = x^3-1 has solutions: ")
print(bisect(lambda x: exp(x**3-1)**0.25, -1, 0, 0.001))
print(bisect(lambda x: exp(x**3-1)**0.25, 1.1, 1.3, 0.001))
print("")

#print(bisect(lambda x: 1/x, -3, 2)) #default tol

"""Given a function f and estimate of the root of f x, true_val returns the error in x.
That is, the distance between the estimate x and the root itself."""
def true_val(f, x):
    return abs(0-f(x))

"""A root finding function based on the secant method. Takes a function f, two initial
guesses x1 & x2 and a tolerance (Default 10**-6)"""
def secant(f, x1, x2, tol=10**-6):  
    #Define a step counter and error term
    steps_counter = 0
    actual_error = true_val(f, x1)

    #While the error is larger than the required error
    while tol < actual_error:
        #Increment the counter
        steps_counter = steps_counter + 1
        
        #Calculate the next iterate
        xn = x1 - f(x1)*((x1-x2)/f(x1)-f(x2))
        
        #Re-index: set appropriate x_(n-1) and x_(n-2) terms
        x1, x2 = xn, x1
        
        #Calculate a new error
        actual_error = true_val(f, xn)

    #Return the root estimate and the number of steps required to reach it
    return (xn, steps_counter)

"""Solve the following using the secand method"""


"""An iterative function to implement Newton's method for approximating the root of a
function given the function f, its derivative df, an initial estimate of the root x0 
and a given tolerence to find the root accurate to (default 10**-6)."""
def newton(f, df, x0, tol=10**-6):
    #Begin counting iterations of the loop/number of times Newton's algorithm is implemented
    steps_counted = 0
    
    #Calculate the error between the root and the initial estimate
    actual_error = true_val(f, x0)
    
    #While the error is larger than the largest acceptable error (tol):
    while tol < actual_error:
        #Apply Newton's algorithm to find a successive approximation to the root
        x0 = x0 - f(x0)/df(x0)
        
        #Increment the step counter
        steps_counted = steps_counted + 1
        
        #Calculate the new actual_error, ready for the loop to either terminate or re-run
        actual_error = true_val(f, x0)
    
    #Return the value of the root, exact to tol. Along with the steps_counted variable
    return (x0, steps_counted)


"""Use Newton's method to solve the following"""
print("Solutions take the form: (root, no. of steps required)")

#h(x) = 4arctan(x) => dh/dx = 4/(x^2+1)
print(newton(lambda x: 4*atan(x), lambda x: 4/(x**2+1), 0, 0.05))
print(newton(lambda x: 4*atan(x), lambda x: 4/(x**2+1), 0.5, 0.05))

#Raises an overflow error: catch this - Inconsistency with fixed point thm
try:
    print(newton(lambda x: 4*atan(x), lambda x: 4/(x**2+1), 1.5, 0.05))
except OverflowError:
    print("Newton's method diverges/is unstable at the initial value x0 = 1.5. |h'(1.5)| > 1 and so this is inconsistent with the fixed point theorem.")

#Raises division by 0 error - Derivative vanishes at x0 = 0
try:
    print(newton(lambda x: x**2-2, lambda x: 2*x, 0, 0.05))
except ZeroDivisionError:
    print("The derivative of h vanishes at x0 = 0. This raises a ZeroDivisonError")

print("")#newline

"""Exercises for feedback"""

#We attempt to find the cube root of 1/2. Sensible initial values are 0 and 1.
print("Finding the root of f(x)=x^3-1/2 by the bisection, secant and Newton methods respectively gives: ")
print("Bisection: " + str(bisect(lambda x: x**3-0.5, 0, 1)[0]))
print("Secant: " + str(secant(lambda x: x**3-0.5, 0, 1)[0]))
print("Newton: " + str(newton(lambda x: x**3-0.5, lambda x: 3*x**2, 0.5)[0]))

#We now run the methods from tolerances 10^-1 to 10^-10
tolerance = 10**-1
steps_taken_bisection, steps_taken_secant, steps_taken_newton = [], [], []

while tolerance >= 10**-10: #Remember it should be >= 
    
    #Add the relevant numbers of steps to relevant tolerance lists.
    steps_taken_bisection.append(bisect(lambda x: x**3-0.5, 0, 1, tolerance)[1])
    steps_taken_secant.append(secant(lambda x: x**3-0.5, 0, 1, tolerance)[1])
    steps_taken_newton.append(newton(lambda x: x**3-0.5, lambda x: 3*x**2, 0.5)[1])

    #Decrement to next tolerance to test/
    tolerance = tolerance / 10

print("") #newline
print("The numbers of steps required to reach these roots for tolerances through 1/10 to 10^-10 is: ")
print("Bisection: " + str(steps_taken_bisection))
print("Secant: " + str(steps_taken_secant))
print("Newton: " + str(steps_taken_newton))

#Finally plot these values
tolerance_list = [10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9, 10**-10]
plt.plot(tolerance_list, steps_taken_bisection, label="Bisection Method")
plt.plot(tolerance_list, steps_taken_secant, label="Secant Method")
plt.plot(tolerance_list, steps_taken_newton, label="Newton Method")

#Update some graphical properties of the plot
plt.title("Plot of no. of steps against tolerance for different root-finding methods")
plt.xlabel("Tolerance")
plt.ylabel("Steps Required")
plt.legend(loc='upper right', ncol=1)
plt.show()