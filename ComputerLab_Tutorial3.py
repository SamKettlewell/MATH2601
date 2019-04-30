# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:06:31 2019

@author: Sam Kettlewell
"""

"""Section 3A - Numpy array exercises. In this section we define various
numpy arrays and matrices and use some of the basic functions associated with
the numpy module."""

"""Import numpy as np"""
import numpy as np

print("Section 3A: Numpy array exercises")
print("")

"""Define the following 4x4 matrices using numpy arrays."""
a = np.ones([4,4])
print(a)
print("")

"""Since a is a numpy array, we can simply multiply every entry in a by 2."""
b= 2*a
print(b)
print("")

"""Define a vector containing integers from 0 to 15 inclusive. Reshape the vector into
a 4x4 matrix."""
c = np.array(range(16))
c = c.reshape([4,4])
print(c)
print("")

"""Print the trace (sum of diagonal entries) of the matrices a, b & c."""
print("trace(a) = " + str(a.trace()))
print("trace(b) = " + str(b.trace()))
print("trace(c) = " + str(c.trace()))
print("")

"""The numpy eye(n, m) function creates a matrix with n rows and m columns. This matrix
has 1s for entries down the main diagonal and 0s everywhere else. If m=n, it produces
the identity matrix Im. The function also behaves senisbly for non-square matrices."""
iden = np.eye(4, 4)
print(iden)
print("")

"""We see by using the det function in the linalg submodule of numpy that the matrices
a, b & c are singular (i.e. have zero determinant). The determinant of c is zero, the
small error is introduced by Python's handling of floating point arithmetic."""
print("det(a) = " + str(np.linalg.det(a)))
print("det(b) = " + str(np.linalg.det(b)))
print("det(c) = " + str(np.linalg.det(c)))
print("")

"""Note that since the matrix c is defined as a numpy array of numpy arrays, we can
simply call sqrt on the entire matrix and it takes the square root componentwise. This
is a very hand feature of numpy."""
d = np.sqrt(c)
print(d)
print("")

"""Use the linalg submodule to find the inverse of the matrix d"""
print(np.linalg.inv(d))
print("")

"""We find the eigenvalues of a and b using the linalg submodule of the numpy
module. It should be noted that the eigenvalues of b are exactly double the
eigenvalues of a as expected. Furthermore, the eigenvalue 0 is repeated three
times (at least to machine accuracy)."""
print("The eigenvalues of a are: " + str([x for x in np.linalg.eig(a)[0]]))
print("The eigenvalues of b are: " + str([x for x in np.linalg.eig(b)[0]]))
print("")

"""In a similar fashion, find the eigenvectors of a and b."""
print("The eigenvectors of a are: \n" + str(np.linalg.eig(a)[1]))
print("The eigenvectors of b are: \n" + str(np.linalg.eig(b)[1]))
print("")


"""3B Slicing Arrays. In this section we import data from a file and find
dimensions, slices and particular elements of matrices."""

print("Section 3B: Slicing Arrays")
print("")

"""Load the data into Python as a matrix"""
q = np.loadtxt("numpy_test.txt")
print("Data laoded into Python successfully.")
print("")

"""Use the shape command to find the dimensions of the matrix q. We see that q
is 10 rows by 7 columns long."""
print("The dimensions of the array q are: " + str(q.shape))
print("")

"""Print the element in row 3, column 2. (Python starts from 0). That is 12167.0"""
print("The element in row 3, column 2 is: " +str(q[3, 2]))
print("")

"""Use the shape command to find the dimensions of the following
submatrix of q."""
print("The dimensions of the submatrix q[2:6, 1:4] of q are: " + str(np.shape(q[2:6, 1:4])))
print("")

"""Use the max function from numpy to find the maximum element of the
following submatrix"""
print("The maximum element of the submatrix q[2:3, :3] of q is: " + str(np.max(q[2:3, :3])))
print("")

"""Print every other element of the first (0th index) column."""
print("Every other element of the first (0th index) column is: \n" + str(q[::2, 0]))
print("")

"""Using the dimensions found above, extract the bottom right 3x3
submatrix."""
print("The bottom right 3x3 submatrix of q is: \n" + str(q[7:10, 4:7]))
print("")



"""Section 3C - Lagrange Interpolation - In this section we define functions
to find the Lagrange basis polynomial and interpolating polynomial of a
function f evaluated at a point x. We then go on to plot the polynomials
using the matplotlib module and fit a Lagrange polynomial to some imported
data."""
#Import the plotting module
import matplotlib.pyplot as plt

print("Section 3C - Lagrange Interpolation")
print("") #newline

"""Given a real number x, a positive integer k and a list of real numbers corresponding
to points at which a function Lk(x) should pass through, return the Lagrange basis
polynomial evaluated at x."""
def lagrange_poly(x, k, xk):
    Lk = 1 #Neutral element w.r.t multiplication
    
    #If k exceeds the number of points, then this makes no sense.
    if k > len(xk):
        return None 
    
    #Iterate through each point given.  
    for i in range(len(xk)):
        
        #Excluding the kth point, multiply the current Lk by the Lagrange
        #basis polynomial evaluated at x.
        if i != (k-1):
            Lk = Lk * ((x - xk[i])/(xk[k-1]- xk[i]))
            
    #Return the value of the Lagrange basis polynomial evaluated at x.
    return Lk

"""Given a real number x, a list of x coordinates and a list of corresponding
y coordinates of n points. This function will compute the value of the
Lagrange interpolating polynomial evaluated at x."""
def interpolating_poly(x, x_list, f_list):
    #If each x coordinate does not have a corresponding y coordinate this makes no sense
    if len(x_list) != len(f_list):
        return None
    
    sum_total = 0 #Neutral element w.r.t addition
    
    #Iterate over each Lagrange Basis Polynomial and eavluate each at x.
    #Add the value of this polynomial to the sum_total.
    for k in range(1, len(x_list)+1):
        sum_total = sum_total + (f_list[k-1] * lagrange_poly(x, k, x_list))
        
    #Return the value of the Lagrange interpolating value 
    return sum_total
        
"""Given a list of x_coordinates and y_coordinates (and some parameters which
correspond to the graphical properties), this function will plot the Lagrange 
Polynomial passing through each (x, y) point. The curve looks 'smooth' (in the
physcial sense, not the differentiable sense) by taking suitably many points
and joining them together."""
def plot_lagrange(x_list, y_list, chart_title, plt_figure, line_label, show_extras = True, show_points = False):
    n = 100 #Generate more points to smoothen the curve.
    
    #If we wish to display the datapoints then plot them.
    if show_points == True:
        plt.figure(plt_figure)
        plt.scatter(x_list, y_list, color='red', label="Data Points")
    
    #Calculate suitable numpy arrays for the domain and range of the interpolating polynomial
    x = np.linspace(min(x_list), max(x_list), len(x_list)*n)    
    
    #Since this is a numpy array, we can just pass it to the function interpolating_poly
    y = interpolating_poly(x, x_list, y_list)
    
    #Choose whether to put the graph on a new chart and plot the polynomial.
    plt.figure(plt_figure)
    plt.plot(x, y, label=line_label)
    
    #Configure axis Labelling Properties
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(chart_title)
    
    #If we wish to display the legend then do so
    if show_extras == True:
        plt.legend(title = "Key", loc = 'upper left')
    
    return None

"""Load the data from lagrange_data.txt and plot the points and the resulting
Lagrange polynomial."""
p = np.loadtxt("lagrange_data.txt")
plot_lagrange(p[:, 0], p[:, 1], "Lagranage Polynomial from Imported Data", 1, "Lagrange Polynomial", show_points=True)
print("See chart entitled 'Lagrange Polynomial from Imported Data' below")
print("")



"""Section 3D - Runge's Phenomenon. In this section we explore Runge's
phenomenon. This is the occurence of large oscillations at the endponts of
the intervals. In particular, we investigate the function f(x) = 1/(1+25x^2)
on the interval [-1, 1]."""
print("Section 3D - Runge's Phenomenon")
print("")

"""Given a positive integer n, this function will obtain n equispaced 
x coordinates on the interval [-1, 1]. Returns as a numpy array."""
def x_runge_coords(n):
    return np.linspace(-1, 1, n)

"""Given a numpy array of x coordinates, this function obtains the 
corresponding y coordinates for the function f(x)=1/(1+25x^2) and returns them
as a numpy array also."""
def y_runge_coords(x_coords):
    return 1/(1+(25*x_coords**2))

"""Iterate through intervals of size n (number of points = n) and plot f(x)
at each of these points. Fit the Lagrange Polynomial. It is very clear to see
that as n grows, the oscillations become more wild. At n=8, they even overpower
the peak of the curve."""
for n in range(2,8):
    #Generate the two numpy arrays of coordinates (x, f(x))
    x_runge = x_runge_coords(n)
    y_runge = y_runge_coords(x_runge)

    #Plot the Lagrange Interpolating Polynomial
    plot_lagrange(x_runge, y_runge, "f(x) = 1/(1+25x²)", 2, "n = " + str(n), show_extras=True, show_points=False)
    
print("See chart entitled 'f(x) = 1/(1+25x^2)' below")
print("")    
    


"""Section 3E - Chebyshev Nodes. In this section we use non-equispaced
x coordinates to try and reduce/remove the large oscillations at endpoints
caused by the Runge Phenomenon in section 3D."""

print("Section 3E - Chebyshev Nodes")
print("")

"""Given a number n, this function returns a numpy array of Chebyshev nodes.
It uses a python list compheresion method to find them all then converts this
list into a numpy array format for ease of calculating the corresponding y
coordinates later."""
def x_runge_chebyshev_nodes(n):
    return np.asarray([np.cos(((2*k-1)*np.pi)/(2*n)) for k in range(1, n+1)])

"""We iterate through intervals of size n and plot f(x) at each point in them.
Also plot -on a seperate chart- the same intervals but each with ten times
as many points in. This allows us to demonstrate how Chebyshev Nodes allow for
a fantastic approximation to the true curve for large values of n. """
for n in [i for i in range(2, 8)]:
    #Calculate the x-coordinate intervals
    x_runge = x_runge_chebyshev_nodes(n)
    tenfold_x_runge = x_runge_chebyshev_nodes(10*n)
    
    #Calculate the corresponding y coordinate arrays
    y_runge = y_runge_coords(x_runge)
    tenfold_y_runge = y_runge_coords(tenfold_x_runge)
    
    #Plot -on two separate graphs- each chart.
    plot_lagrange(x_runge, y_runge, "f(x) = 1/(1+25x²) - Chebyshev Nodes - Small Values", 3, "n = " + str(n), show_extras=True, show_points=False)
    plot_lagrange(tenfold_x_runge, tenfold_y_runge, "f(x) = 1/(1+25x²) - Chebyshev Nodes - Large Values", 4, "n = " + str(10*n), show_extras=True, show_points=False)
    
print("See charts entitled 'f(x) = 1/(1+25x^2)  Chebyshev Nodes - Small Values' and 'f(x) = 1/(1+25x^2)  Chebyshev Nodes - Large Values' below")