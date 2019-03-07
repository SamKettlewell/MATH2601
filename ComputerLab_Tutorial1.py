# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 13:10:26 2019

@author: Sam Kettlewell
"""
from math import exp, factorial, gcd, floor, sqrt       

class Rational:
    def __init__(self, num, den):
        """Constructor."""
        hcf = gcd(num, den)
        self.num = num // hcf
        self.den = den // hcf

    def __repr__(self):
        """Return string representation of a rational."""
        return str(self.num) + ' / ' + str(self.den)

    def __mul__(self, rhs):
        """Return product of two rationals."""
        new_num = self.num * rhs.num
        new_den = self.den * rhs.den
        return Rational(new_num, new_den)

    def __add__(self, rhs):
        """Return the sum of `self` and `rhs`."""
        new_num = (self.num * rhs.den) + (self.den * rhs.num)
        new_den = self.den * rhs.den
        return Rational(new_num, new_den)

    def __neg__(self):
        """Return the negative of `self`."""
        new_num = -1 * self.num
        return Rational(new_num, self.den)

    def __sub__(self, rhs):
        """Return the result of `self` minus `rhs`."""
        #(a/b) - (c/d) is the same as (a/b) + (-c/d)
        #Rational(self.num, self.den) + Rational(rhs.num, rhs.den).__neg__()
        
        new_num = (self.num * rhs.den) - (self.den * rhs.num) #Using similar algebra to __add__
        new_den = self.den * rhs.den
        return Rational(new_num, new_den)

    def __eq__(self, rhs):
        """ This function implements the == operator which tests for equality.
        Two fractions are equal if their numerator and denominator are the same. """
        return self.num == rhs.num and self.den == rhs.den

    def __gt__(self, rhs):
        """ This function implements the > operator.
        A fraction is bigger than another fraction if their difference is
        positive, which is the case if the numerator of the difference is > 0. """
        difference = self - rhs
        return difference.num > 0

    def floor(self):
        """Return `self` rounded down to nearest integer."""
        return self.num // self.den

    def frac(self):
        """Return fractional part of `self`."""
        return self - Rational(self.floor(), 1)

    def to_float(self):
        """Return `self` converted to a floating-point number."""
        return self.num / self.den
    
    def expand(self, base, n):
        if self.num < 0:
            self = -self
            res = '-'
        else:
            res = ''
        left = self.expand_integer_part(base)
        right = self.expand_fractional_part(base, n)
        for digit in left:
            res += str(digit)
        res += '.'
        for digit in right:
            res += str(digit)
        res += ' (base ' + str(base) + ')'
        return res

    def expand_fractional_part(self, base, n):
        r = self.frac()
        rational_base = Rational(base,1)
        digits = []
        while len(digits) < n and r.num != 0:
            base_times_r = rational_base * r
            a = base_times_r.floor()
            r = base_times_r.frac()
            digits.append(a)
        return digits
    
    def expand_integer_part(self, base):
        integer_part = self.floor()
        if integer_part == 0:
            return [0]
        if integer_part > 0:
            sign = 1
        else:
            sign = -1
            integer_part *= -1
        digits = []
        while integer_part > 0:
            digits.append(sign * (integer_part % base))
            integer_part //= base
        digits.reverse()
        return digits

################################################################################
###############################-Main body of code-##############################
################################################################################
    
print(exp(1)) #accurate to 15 d.p.

"""Function to return a decimal approximation to exp(x) using Taylor series' """
def exp_taylor_decimal(x, n):
    exp_approx = 0
    
    for i in range(n):
        exp_approx += x**i / factorial(i)
    
    return exp_approx

"""Function to return a rational approximation to exp(x) using Taylor series' and
the Rational class to handle fractions."""
def exp_taylor_rational(x, n):
    exp_approx = Rational(1,1) #Defines the term when n=0 so must begin from 1
    
    for i in range(1, n+1):
        #Since division is not supported, we multiply by 1/reciprocol
        one_over_n_fact = Rational(1, factorial(i))
        exp_approx += Rational(x**i, 1) * one_over_n_fact
    
    return exp_approx

"""Going up to 18 terms in the Taylor Series, compute and display each sum in a rational
and decimal format."""
for a in range(18):
    print(exp_taylor_decimal(1, (a+1))) #Start from 1 since the first rational approximant is 1/1
    print(exp_taylor_rational(1, a)) #18 iterations are required to reach 15 d.p accuracy
    print("") #newline

"""Testing how the expand function behaves: it expands any rational as a decimal
in base n to m d.p. for m,n arguments of the expand function."""
one_seventh = Rational(1, 7)
print(one_seventh.expand(10, 20)) #NB 1/7 is a recurring decimal so we can expand it infinitely

"""To locate the 100th decimal place of e we use the above Rational taylor expansion
of e^x."""
e_approx = exp_taylor_rational(1, 2500)
print(e_approx.expand_fractional_part(10, 100)[99]) #The 100th digit is 4? This disagrees with online?

"""Function to approximate pi using Euler's approximation"""
def approx_pi(n):
    pi_approx = Rational(1, 1) #first term in series
    
    for k in range(1, n+1): #loop through rest of terms
        numerator = (2**k) * (factorial(k)**2) #euler's approximation
        denominator = factorial(2*k  + 1)
        
        pi_approx += Rational(numerator, denominator) #add next term in series
        
    return Rational(2, 1) * pi_approx

pi_approximation = approx_pi(165) #At least 165 terms required to be accurate to 50 d.p.
print("The 50 d.p of pi is: " + str(pi_approximation.expand_fractional_part(10, 50)[-1]))

"""The double factorial is used the in the Ramanujan approximation"""
def double_factorial(k):
    if k%2 == 0: #Only applies to odd numbers
        return False
    
    product = 1
    for num in range(1, k+1, 2): #Iteratively calculate the product of odd numbers
        product = product * num
    
    return product

"""Function to approximate pi using Ramanujan's series"""
def ramanujans_approximation(n):
    pi_approx = Rational(1123, 882) #Fantastic first approximation btw
    
    for k in range(1, n+1): #loop through the terms
        numerator = (-1)**k * (1123 + 21460*k) * double_factorial(2*k - 1) * double_factorial(4*k - 1)
        denominator = (882**(2*k + 1)) * (32**k) * factorial(k)**3
        
        pi_approx += Rational(numerator, denominator) #add successive terms of the series
    
    pi_approx = Rational(pi_approx.den, pi_approx.num) #take the reciprocol
    
    return Rational(4, 1) * pi_approx
    
pi_approximation = ramanujans_approximation(8) #Need at least 8 terms to guarentee correctness to 50 d.p.
print("The 50th digit of pi is: " + str(pi_approximation.expand_fractional_part(10, 50)[-1]))

"""ADVANCED"""
"""This function -the backbone of which was salvaged from pre-existing MATH2920 code
creates a list corresponding to a continued fraction."""
def continued_fraction(x, i):
    cont_frac_list = []   
    rational_approximation = Rational(1, 2) #How do I adapt this for ANY cont. frac?
    
    #make a list of the continued fraction - stolen from math2920 module code
    while i > 0:
        cont_frac_list.append(floor(x))
        x = 1/(x-floor(x))
        i -= 1
    
    cont_frac_list.reverse()
    
    #Loop through the list in reverse and build successive rational approximations
    for each_term in cont_frac_list:
        rational_approximation = rational_approximation + Rational(each_term, 1)
        rational_approximation = Rational(rational_approximation.den, rational_approximation.num)
    
    #Take reciprocol
    rational_approximation = Rational(rational_approximation.den, rational_approximation.num)

    #Return both the list and the approximation in a tuple
    return (cont_frac_list, rational_approximation)

"""Ramanuan's second approximation reuqires far fewer terms to approximate pi. This is
an insanely accurate approximation!"""
def ramanujans_approximation2(n):
    pi_approx = Rational(1103, 1) #First approximation
    
    #Loop through n terms and sum successive terms
    for k in range(1, n+1):
        numerator = factorial(4*k) * (1103 + 26390*k)
        denominator = factorial(k)**4 * 396**(4*k)
        
        pi_approx += Rational(numerator, denominator)
        
    sqrt_2 = continued_fraction(sqrt(2), 20)[1] #Rational approx to sqrt(2), to 20 terms
    pi_approx = pi_approx * Rational(2, 9801) #Mutliply by a suitable constant
    pi_approx = Rational(pi_approx.den, pi_approx.num) #Take reciprocol
    
    #Got to divide through by sqrt(2) here. This is equivalent to multiplying by 1/sqrt2
    sqrt_2 = Rational(sqrt_2.den, sqrt_2.num)
    pi_approx = pi_approx * sqrt_2
    
    return pi_approx

#Question: It feels like I've been a bit sloppy with the reciprocals. Is there a better
#way to take them? Should I have implemented it as a method in the Rational class?