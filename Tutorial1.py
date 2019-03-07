# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:07:31 2019

@author: Sam Kettlewell
"""

from math import gcd

class Rational:
    def __init__(self, num, den):
        hcf = gcd(num, den) #Reduces the fraction to lowest terms
        self.num = num // hcf
        self.den = den // hcf
        
    def display(self):
        print(str(self.num) + '/' + str(self.den)) #?
        
#two_thirds = Rational(2,3)
#print(two_thirds.den) #access attributes of Rational
#two_thirds.display() #call a method of Rational?

print("")

twelve_sixteenths = Rational(12, 16) #This can be simplified
twelve_sixteenths.display()

class Rational_v2:
    def __init__(self, num, den):
        hcf = gcd(num, den) #Reduces the fraction to lowest terms
        self.num = num // hcf
        self.den = den // hcf
        
    def __repr__(self): #Instructions for printing an instance of Rational_v2
        return str(self.num) + '/' + str(self.den) #+ "blahblah"
    
three_quarters = Rational_v2(6,8)
print(three_quarters)

class Rational_v3:
    def __init__(self, num, den):
        hcf = gcd(num, den) #Reduces the fraction to lowest terms
        self.num = num // hcf
        self.den = den // hcf
        
    def __repr__(self): #Instructions for printing an instance of Rational_v3
        return str(self.num) + '/' + str(self.den)# + "blahblah"
    
    def __mul__(self, rhs):
        new_num = self.num * rhs.num
        new_den = self.den * rhs.den
        
        return Rational_v3(new_num, new_den) #this will also reduce to lowest terms bc v1
    
fifteen_twentififths = Rational_v3(15, 25)
two_thirds = Rational_v3(2, 3)
print(fifteen_twentififths * two_thirds)

class Rational_v4:
    def __init__(self, num, den):
        hcf = gcd(num, den) #Reduces the fraction to lowest terms
        self.num = num // hcf
        self.den = den // hcf
        
    def __repr__(self): #Instructions for printing an instance of Rational_v4
        return str(self.num) + '/' + str(self.den)# + "blahblah"
    
    def __mul__(self, rhs):
        new_num = self.num * rhs.num
        new_den = self.den * rhs.den
        
        return Rational_v3(new_num, new_den) #this will also reduce to lowest terms bc v1

    def to_float(self):
        return self.num / self.den
    
improper_twelve_ninths = Rational_v4(12,9)#method to tell whether improper?
print(improper_twelve_ninths.to_float())