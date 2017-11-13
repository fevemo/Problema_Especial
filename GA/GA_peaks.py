import numpy as np
import matplotlib.pyplot as plt

###CONSTANTS###
N=20
xRange=[-3,3]
yRange=[-3,3]
zero='0000000000000000'
l_max=len(bin(int(pow(2,16)-1))[2:])

###Objective function###
def f(x,y):
    return 3*((1-x)**2)*(np.exp(-x**2-(y+1)**2))-10*(x/5-x**3-y**5)*np.exp(-x**2-y**2)-np.exp(-(x+1)**2-y**2)/3

###Converts decimal numbers in range (xRange, yRange) to binary representations###
def encode(x,y):
    x,y=transScale(x,y)
    encode=[]
    for i in range(len(x)):
        x_i=bin(int(x[i]))[2:]
        y_i=bin(int(y[i]))[2:]
        l_x=len(x_i)
        l_y=len(y_i)
        x_i=zero[:(l_max-l_x)]+x_i
        y_i=zero[:(l_max-l_y)]+y_i
        encode.append(x_i+y_i)
    return encode

###Translates and scales numbers in range (xRange, yRange) to int range (0, 2E16-1)###
def transScale(x,y):
    x=(x-xRange[0])/(xRange[1]-xRange[0])*(pow(2,16)-1)
    y=(y-yRange[0])/(yRange[1]-yRange[0])*(pow(2,16)-1)
    return x,y

###Converts binary numbers to decimal by doing the inverse of encode(x,y)###
def decode(e):
    x=np.zeros(N)
    y=np.zeros(N)
    for i in range(N):
        x[i]=int(e[i][:16],2)
        y[i]=int(e[i][16:],2)
        
    return invTransScale(x,y)

###Translates and scales numbers in range (0, 2E16-1) to range (xRange, yRange)###
def invTransScale(x,y):
    x=(x/(pow(2,16)-1)*(xRange[1]-xRange[0]))+xRange[0]
    y=(y/(pow(2,16)-1)*(yRange[1]-yRange[0]))+yRange[0]
    return x,y

###Randomly generates an initial population of size N###
def P0():
    x=6*np.random.random_sample(N)-3
    y=6*np.random.random_sample(N)-3
    return x, y

### Returns the best specimen S and f(S) (Pos(s),f(s))###
def find_max(x, y):
    max=-1E50
    i_max=-1
    for i in range(len(x)):
        f_i=f(x[i],y[i])
        if (f_i>max):
            max=f_i
            i_max=i
    return i_max,max

def sel_tournament(x, y):
    for i in range(N):
        i_a=np.random.randint(0,N)
        i_b=np.random.randint(0,N)
        
        f_a=f(x[i_a],y[i_a])
        f_b=f(x[i_b],y[i_b])
        
        if(f_a>=f_b):
            x[i]=x[i_a]
            y[i]=y[i_a]
        else:
            x[i]=x[i_b]
            y[i]=y[i_b]
            
    return x,y

def sel_prob(x,y):
    indices=[]
    x_n=np.zeros(N)
    y_n=np.zeros(N)
    for i in range(N):
        p=int(np.max([f(x[i],y[i]),0]))
        indices=indices+np.ndarray.tolist((i*np.ones(p)))
    print(indices)
    for i in range(N):
        i_s=indices[np.random.randint(0,len(indices))]
        x_n[i]=x[i_s]
        y_n[i]=y[i_s]
    return x_n, y_n

def crossover(e):
    p_c=0.8
    k=int(np.random.normal(p_c*N))
#    print(k)
    for i in range(k):
        i_a=np.random.randint(0,N)
        i_b=np.random.randint(0,N)
        
        pom=np.random.randint(0,l_max*2)
#        print(e[i_a])
        e[i_a]=e[i_a][:pom]+e[i_b][pom:]
        e[i_b]=e[i_b][:pom]+e[i_a][pom:]
#        print(e[i_a])
    return e

def mutation(e):
    p_m=0.3
    for i in range(N):
        if(np.random.random()<p_m):
#            print('mutation')
            p=np.random.randint(0,l_max*2)
            if(e[i][p]==0):
                e[i]=e[i][:p]+'1'+e[i][p+1:]
            else:
                e[i]=e[i][:p]+'0'+e[i][p+1:]
    return e

def grade(x,y):
    return np.average(f(x,y))
###Testing###
J=20
max_ga=[]
avg_ga=[]
max_rand=[]
for j in range(J):
    I=50
    x,y=P0()
    max_evol=np.zeros(I)
    grade_evol=np.zeros(I)
    bsf_max=-1E50
    bsf_x=0
    bsf_y=0
    for i in range(I):
        cs=find_max(x,y)
        max_evol[i]=cs[1]
        grade_evol[i]=grade(x,y)
        if(cs[1]>bsf_max):
            bsf_x=x[cs[0]]
            bsf_y=y[cs[0]]
            bsf_max=cs[1]
        x_s,y_s=sel_tournament(x,y)
        e_c=crossover(encode(x_s,y_s))
        e_m=mutation(e_c)
        x,y=decode(e_m)
#    print(x,y)
    max_ga.append(bsf_max)
    avg_ga.append(grade(x,y))
print("Best solution: (",bsf_x,",",bsf_y,") with max: ",bsf_max)
#plt.plot(max_evol)
#plt.plot(grade_evol)
#plt.ylim([0,8.15])
#plt.show()
    
x_r=6*np.random.random_sample(N*I)-3
y_r=6*np.random.random_sample(N*I)-3
max_rand.append(find_max(x_r,y_r)[1])

plt.plot(max_ga,label='Genetic Algorithm (max)')
plt.plot(avg_ga,label='Genetic Algorithm (avg)')
plt.plot(max_rand,label='Random sample')
plt.legend()
plt.show()
#


