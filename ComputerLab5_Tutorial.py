# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:54:15 2019

@author: Sam Kettlewell
"""

"""PS: The enitre script took 15 seconds to execute on my computer. I hope this isn't too long/inefficient. If it is, feel free to comment out some
of the plots. The Butterfly_System one will probably conserve a lot of time if commented out."""

#Import relevant modules
from matplotlib import pyplot as plt
import numpy as np

"""Exercise 5A - The Euler & Runge-Kutta Methods"""
#Define the autonomous ODE f(x, t):= dx/dt = x.
def f(x):
    return x

"""Do these only have to be for autonomous equations?!?! x'=x"""

#Define a function 'euler' which implements the forward Euler method for autonomous ODEs
def euler(x, f, h):
    return x + f(x) * h

#Define a function: 'modified euler' which implements the modified euler method for autonomous ODEs
def modified_euler(x, f, h):
    k1 = h * f(x)
    k2 = h * f(x + k1)
    return x + 0.5 * (k1+k2)

#Define the classic 'fourth order Runge Kutta' method  which implements the RK4 method for autonomous ODEs
def fourth_order_RungeKutta(x, f, h):
    k1 = h * f(x)
    k2 = h * f(x + h/2) #Technically we don't need this in two variables but I'll keep it for mathematical consistency.
    k3 = h * f(x + h/2)
    k4 = h * f(x + h)
    
    return x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)

"""Define a function, plot_ODE_solutions, which when given an integrator, time-step, and various graphical properties, should return
a plot of the approximation to the ODE using the supplied integrator."""
def plot_ODE_solutions(num_scheme, h, title, label, fig_num):
    #Need to work error here.
    
    #Calculate the coordinates of the numerical integrators for 0 <= t <= 5.
    x = 1
    x_values, t_values, errors = [], [], []
    for i in range(5*int(1/h)+1): #Reach t=5
        error = np.abs(np.exp(i*h)-x)
        x = num_scheme(x, f, h)
        t_values.append(i*h)
        x_values.append(x)
        errors.append(error)
        #print("x({:5.2f}) = {:7.4f}".format(i*h, x)) #To print them to the console for debugging purposes.
    
    #Configure any graphical properties and plot the coordinates calculated above.
    plt.figure(fig_num)
    plt.title(title)
    plt.xlabel("$t$")
    plt.ylabel("$x$")
    plt.plot(t_values, x_values, label=label)
    
    return None

"""Plot a graph of all three numerical integrating schemes for timestep h=1. Overlay the true solution y(t) = exp(t)."""
title = "Timestep h = 1"

plot_ODE_solutions(euler, 1, title, "Euler", 1)
plot_ODE_solutions(modified_euler, 1, title, "Modified Euler", 1)
plot_ODE_solutions(fourth_order_RungeKutta, 1, title, "Runge Kutta", 1)

plt.plot(np.linspace(0, 5, 100), np.exp(np.linspace(0, 5, 100)), label = "y(t)=exp(t)") #True solution
plt.legend()

"""Plot a (new) graph of all three numerical integrating schemes for timestep h=0.1. Overlay the true solution y(t) = exp(t)."""
title = "Timestep h = 0.1"

plot_ODE_solutions(euler, 0.1, title, "Euler", 2)
plot_ODE_solutions(modified_euler, 0.1, title, "Modified Euler", 2)
plot_ODE_solutions(fourth_order_RungeKutta, 0.1, title, "Runge Kutta", 2)

plt.plot(np.linspace(0, 5, 100), np.exp(np.linspace(0, 5, 100)), label = "y(t)=exp(t)") #True solution
plt.legend()

"""Plot a (new) graph of all three numerical integrating schemes for timestep h=0.01. Overlay the true solution y(t) = exp(t)."""
title = "Timestep h = 0.01"

plot_ODE_solutions(euler, 0.01, title, "Euler", 3)
plot_ODE_solutions(modified_euler, 0.01, title, "Modified Euler", 3)
plot_ODE_solutions(fourth_order_RungeKutta, 0.01, title, "Runge Kutta", 3)

plt.plot(np.linspace(0, 5, 100), np.exp(np.linspace(0, 5, 100)), label = "y(t)=exp(t)") #True solution
plt.legend()

"""Observe that, as h decreases, the accuracy of the numerical integrating scheme increases. You can see this by observing that on
each successive graph where h decreases tenfold, all the numerical approximations to the integral become closer (converge uniformly?)
to the curve corresponding to the true solution y(t) = exp(t). This is, however, a trade-off since a smaller h value requires more 
calculations to be performed to reach the same maximum t-value as in previous numerical integration schemes. This small h value is
easy to work with computationally but would be a pain to work with on pen and paper."""




"""Exercise 5B/5C - The n body problem"""

"""Define a function, drdt, which takes a Python list of numpy arrays corresponding to planetary velocities and returns a tuple
containing each velocity vector in the array."""
def drdt(v):
    return tuple(v)

"""Define a function, dvdt, which takes a Python list of numpy arrays corresponding to planetary position vecotrs and returns a tuple
containing ..."""
def dvdt(r, Gm_array):
    
    """Iterate through all the position vectors and produce a list containing all vectors which the accelerations are projected along
    for n bodies. (i.e. for n=3, produce a list of numpy arrays containing r12 = r2 - r1; r13 = r3 - r1; r23 = r3 - r2.)"""
    #Define empty lists to store each projection vector and its corresponding norm.
    vd, n = [], []
    index1, index2 = 0, 0
    while index1 < len(r):
        while index2 < len(r):
            vector  = r[index2] - r[index1] #Vector that acceleration is projected along accoring to Newton's Law of Gravitaition
            vd.append(vector)
            n.append(np.linalg.norm(vector)) #Norm of vector
            index2 += 1   
        index1 += 1
        index2 = 0

    """Iterating through each vector in the list above, calculate all the relevant acceleration components and sum them to produce a list of accelerations."""
    a = []
    index1, index2 = 0, 0
    while index1 < len(r):
        s=0
        while index2 < len(r):
            if index1 != index2: #prevent division by 0 nan errors
                access = len(r)*index1 + index2
                component = Gm_array[index2] * vd[access] / n[access] ** 3 #Need to modify this Gm access list
                s = s + component       
            index2 += 1     
        a.append(s) 
        index1 += 1
        index2 = 0
        
    #Return a tuple containing all planetary accelerations
    return tuple(a)


"""Define an integrator function, euler_simplectic, which takes an array of position and velocity vectors (and an array corresponding to Gm) and
returns a tuple of arrays of position and velocity vectors (in that order) after numerical integration by the Euler Simplectic scheme."""
def euler_simplectic(pos_array, vel_array, h, Gm_array):
    #Produce a tuple of accelerations for use in the next integration
    accel_tuple = dvdt(pos_array, Gm_array)
    
    #Rewrite the velocity array
    for acceleration_index in range(len(accel_tuple)):
        new_vel = vel_array[acceleration_index] + h * accel_tuple[acceleration_index]
        vel_array[acceleration_index] = new_vel
    
    #Produce an array of velocities ???
    vel_tuple = drdt(vel_array)
    
    #Rewrite the position array
    for velocity_index in range(len(vel_tuple)):
        new_pos = pos_array[velocity_index] + h * vel_tuple[velocity_index]
        pos_array[velocity_index] = new_pos
        
    return pos_array, vel_array





"""Section 5B/5C - Improving the code - I will start by constructing a class Planet_System (task 5C/(d)) and then use this class to demonstrate various 
solutions to the three body problem and one plot of our solar system (a solution to the 10 body problem)."""
class Planet_System:
    
    """Initialise the Planet System's associated variables in such a way that they can be accessed by calling self."""
    def __init__(self, name, init_pos, init_vel, Gm):
        
        #Initialise the variables passed to the instance of Planet_System
        self.name = name #String
        self.init_pos = init_pos #Numpy array of numpy arrays
        self.init_vel = init_vel #Numpy array
        self.Gm = Gm #Numpy array
        
        return None
    
    """Define a function, static_plot, taking arguments corresponding to time-step, number of calculations and any graphical properties and produce 
    a still, 2D plot of the planetary system trajectory in the x-y plane."""
    def static_plot(self, nstep, h, fig_no):
        t=0 #Time evolution parameter
        
        #Generate a list of position and velocity vectors. (And time coordinates)
        r = [[pos for pos in self.init_pos]]
        v = [[vel for vel in self.init_vel]]
        time = [t]
        
        #Perform the integration using the Euler Simplectic numerical integration scheme.
        for i in range(nstep):
            #Calculate the 'next' spatial coordinates and velocity.
            r_new, v_new = euler_simplectic(r[-1][:],v[-1][:],h, self.Gm) #It is important to take slices/a fresh copy of the whole list each time.
            
            #Add the new vectors corresponding to the planet coordinates and velocity in each step to the lists r and v.
            r.append(r_new)
            v.append(v_new)
            
            #Increment the discretized timestep
            t = t + h
            time.append(t)
        
        #Calculate the number of planets in the system
        planets_in_system = len(r[0])
        
        #Put the position coordinate list into a numpy array - Allows easy access
        R=np.array(r)
        
        #Start a new chart (if appropriate)
        plt.figure(fig_no)
        
        """Plot each planet's trajectory for each planet in the system in the xy plane."""
        for each_planet in range(planets_in_system):
            plt.plot(R[:,each_planet,:][:,0], R[:,each_planet,:][:,1])
            
        #Give the graph a title
        plt.title("Planet System: " + str(self.name))
        
        return None
        
    #The representation of the planet.
    def __repr__(self):
        return "Planet System : " + self.name + "\nNo. of Planets: " + str(len(self.init_pos))

"""Exercise 5B/5C(d) - Generate a figure of eight pattern. I have improved my code using the class Planet_System"""

a, b = 0.347111, 0.532728
Figure_of_eight_System = Planet_System("Figure of Eight System (3 Body Problem)", np.array([[-1,0,0], [1,0,0], [0,0,0]]), np.array([[a,b,0], [a,b,0], [-2*a, -2*b, 0]]), np.array([1,1,1]))
Figure_of_eight_System.static_plot(3000, 0.01, 4)

"""Exercise 5C(a)/5C(d) - Create some instances of different three body Systems. I have improved my code using the class Planet_System"""

#This system is shaped like a Butterfly. It is more difficult/costly to generate than any other plot due to the small time-step required.
a, b = 0.306893, 0.125507
Butterfly_System = Planet_System("Butterfly System", np.array([[-1,0,0], [1,0,0], [0,0,0]]), np.array([[a,b,0], [a,b,0], [-2*a, -2*b, 0]]), np.array([1,1,1]))
Butterfly_System.static_plot(63000, 0.0001, 5)

""" 
#Omitted due to time taken to produce. If you have a powerful PC or want to see what is (meant to be) some goggles then uncomment this.


#This system is (apparently) shaped like a pair of goggles. It is about as difficult/costly to generate as the Butterfly System.
a, b = 0.083300, 0.127889
Goggle_System = Planet_System("Goggle System", np.array([[-1,0,0], [1,0,0], [0,0,0]]), np.array([[a,b,0], [a,b,0], [-2*a, -2*b, 0]]), np.array([1,1,1]))
Goggle_System.static_plot(60000, 0.0001, 13)
"""

#This system is (meant to be) shaped like a Dragonfly (I struggle to see it personally). It is quite difficult/costly to generate
#I had to choose the second set of a,b values in this family because the first set were too costly to generate in a reasonable time.
a, b = 0.186238, 0.578714
Dragonfly_System = Planet_System("Dragonfly System", np.array([[-1,0,0], [1,0,0], [0,0,0]]), np.array([[a,b,0], [a,b,0], [-2*a, -2*b, 0]]), np.array([1,1,1]))
Dragonfly_System.static_plot(20000, 0.001, 6)
#Perhaps the system would look nicer with a smaller h-value but I don't have the computational power to demonstrate that.

#Aside from the figure of eight system, the Yarn System was the easiest to generate. It made use of a 'relatively' large timestep value
#and not many steps (relative, once again) were required to plot the trajectory.
a, b = 0.464445, 0.396060
Yarn_System = Planet_System("Yarn/Moth System", np.array([[-1,0,0], [1,0,0], [0,0,0]]), np.array([[a,b,0], [a,b,0], [-2*a, -2*b, 0]]), np.array([1,1,1]))
Yarn_System.static_plot(15000, 0.001, 7)

"""Here, I demonstrate that the code is robust enough to solve the n=2 body problem also."""

TwoD_System = Planet_System("Two Dimensional System", np.array([[1,0,0], [-1,0,0]]), np.array([[0,0.01,0], [0,-2,0]]), np.array([10,0.05]))
TwoD_System.static_plot(2000, 0.01, 8)

"""Exercise 5C(b)/5C(d) - Generate a plot of our Solar System. I have improved my code using the class Planet_System. I have chosen not 
to plot the entire trajectory of the planets with larger orbital radii as this would make the code take too long to execute."""

#Import data from an external Python file corresponding to our solar system and reshape it for appropriate use with the above functions
import solardat

#Planetary Spatial Coordinates: (x,y,z)
solar_system_init_pos = solardat.r
solar_system_init_pos = solar_system_init_pos.reshape([10,3])

#Planetary Velocity Coordinates: (vx,vy,vz)
solar_system_init_vel = solardat.v
solar_system_init_vel = solar_system_init_vel.reshape([10,3])

#Solar System Gm array
solar_system_Gm = solardat.Gm

#Create an instance of our Solar System and plot it on a new figure.
Solar_System = Planet_System("Solar System", solar_system_init_pos, solar_system_init_vel, solar_system_Gm)
Solar_System.static_plot(5000, 1, 9)

"""Exercise 5C(c) - Perform a numerical analysis on the ODE f(x):= x' = x ."""

"""Define a function, similar to plot_ODE_solutions above, which calculates the error at each time between t=0 and t=t (an argument passed to the function).
Plot this """
def error_until_time(num_scheme, f, h, t, fig_num, title, label):

    #Calculate the errors
    x = 1
    errors, t_list = [], []
    for i in range(t*int(1/h)+1): #Reach t=5
        x = num_scheme(x, f, h)
        t_list.append(i*h)
        error = np.abs(np.exp(i*h)-x)
        errors.append(error)
    
    #Plot the figure
    plt.figure(fig_num)
    plt.plot(t_list, errors, label=label)
    plt.title(title)
    plt.legend()
    
    return None


"""Plot a graph of the errors all three numerical integrating schemes for timestep h=1."""
title = "Error: Timestep h = 1"

error_until_time(euler, f, 1, 5, 10, title, "Euler Error")
error_until_time(modified_euler, f, 1, 5, 10, title, "Modified Euler Error")
error_until_time(fourth_order_RungeKutta, f, 1, 5, 10, title, "Runge Kutta Error")

"""Plot a graph of the errors all three numerical integrating schemes for timestep h=0.1."""
title = "Error: Timestep h = 0.1"

error_until_time(euler, f, 0.1, 5, 11, title, "Euler Error")
error_until_time(modified_euler, f, 0.1, 5, 11, title, "Modified Euler Error")
error_until_time(fourth_order_RungeKutta, f, 0.1, 5, 11, title, "Runge Kutta Error")

"""Plot a graph of the errors all three numerical integrating schemes for timestep h=0.01."""
title = "Error: Timestep h = 0.01"

error_until_time(euler, f, 0.01, 5, 12, title, "Euler Error")
error_until_time(modified_euler, f, 0.01, 5, 12, title, "Modified Euler Error")
error_until_time(fourth_order_RungeKutta, f, 0.01, 5, 12, title, "Runge Kutta Error")

#Observation: Where the approximation crosses the true solution, the error curve is 0.