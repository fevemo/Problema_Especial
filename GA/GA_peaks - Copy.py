import numpy as np
import matplotlib.pyplot as plt

xRange=[-3,3]
yRange=[-3,3]

#Objective function
def f(x,y):
    return 3*((1-x)**2)*(np.exp(-x**2-(y+1)**2))-10*(x/5-x**3-y**5)*np.exp(-x**2-y**2)-np.exp(-(x+1)**2-y**2)/3

#Converts decimal numbers in range (xRange, yRange) to binary representations
def encode(x,y):
    x,y=transScale(x,y)
    return bin(x)[2:],bin(y)[2:]

def transScale(x,y):
    x=(x-xRange[0])/(xRange[1]-xRange[0])*(2E16-1)
    y=(y-yRange[0])/(yRange[1]-yRange[0])*(2E16-1)
    return int(x),int(y)

#Converts binary numbers to decimal by doing the inverse of encode(x,y)
def decode(x,y):
    x=int(x,2)
    y=int(y,2)
    return invTransScale(x,y)

def invTransScale(x,y):
    x=(x/(2E16-1)*(xRange[1]-xRange[0]))+xRange[0]
    y=(y/(2E16-1)*(yRange[1]-yRange[0]))+yRange[0]
    return x,y







#Testing
#a=encode(-2.8,3)
#print(a)
#b=decode(a[0],a[1])
#print(b)
#print(len(a[0]))
#print(f(-0.0093,1.5814))
#print(f(-3,3))
