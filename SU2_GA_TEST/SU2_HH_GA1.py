import numpy as np
import matplotlib.pyplot as plt
import os, sys, shutil, copy
sys.path.append(os.environ['SU2_RUN'])
import SU2
from optparse import OptionParser
from joblib import Parallel, delayed

#Number of Hicks-Henne functions that will deform the mesh
L=10
#Number of specimens in each generation
N=10
#Minimum and maximum values of DV_VALUE allowed
mini=-0.01
maxi=0.01

#SU2 config variables
nPart=8
file='inv_NACA0012_GA.cfg'
OG_MESH=''

def main():
    # Load configuration file
    parser=OptionParser()
    parser.add_option("-f", "--file",       dest="filename",
                      help="read config from FILE", metavar="FILE")
    parser.add_option("-n", "--partitions", dest="partitions", default=8,
                      help="number of PARTITIONS", metavar="PARTITIONS")

    (options, args)=parser.parse_args()
    nPart = int( options.partitions )
    if options.filename == None:
        raise Exception("No config file provided. Use -f flag")
    else:
        file=options.filename
        
    config=setupSU2()
    GA(config)
         
        
def GA(config,K=30,p_c=0.5,p_m=0.1):
    P=gen_population()
    avgFit=np.zeros(K)
    maxFit=np.zeros(K)
    fit_bsf=-1E20
    bsf=[]
    for k in range(K):
	print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
	print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")        
	print("evaluating population: ",k)
	print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
	print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        fit=eval_pop(config,P)
        avgFit[k]=sum(fit)/len(fit)
        max_fit_k=max(fit)
        maxFit[k]=max_fit_k
        if(max_fit_k>fit_bsf):
            fit_bsf=max_fit_k
            bsf=P[np.argmax(fit)]
    
        P=mutation(p_m,combination_single_cross(p_c,selection_tournament(fit,P)))
    
    print('Best solution: ',bsf,' with score: ', fit_bsf)
    
    plt.plot(avgFit,label='Average')
    plt.plot(maxFit,label='Maximum')
    plt.legend()
    plt.show()
         
    

def setupSU2():        
    config=SU2.io.Config(file)
    print(file)
    config.NUMBER_PART = 8
    config.CONSOLE = 'CONCISE'
    #config.DEFORM_CONSOLE_OUTPUT='NO'
    OG_MESH='mesh_NACA0012_inv.su2'#config.MESH_FILENAME
#    kind_array=[]
#    value_array=[]
#    for i in range(L):
#        kind_array.append('HICKS_HENNE')
#        value_array.append(0)
#    config.DV_KIND=kind_array
#    config.DV_VALUE=value_array
    return config

#Generate new mesh and set it as current mesh
def deform(config, new_DV):
    konfig=copy.deepcopy(config)
    print(new_DV)
    #Setting new design variables
    #konfig.MESH_FILENAME=OG_MESH
    #print('mesh name: '+konfig.MESH_FILENAME)
    konfig.DV_VALUE_NEW=new_DV
    konfig.MESH_OUT_FILENAME='mesh.su2'
    
    SU2.run.DEF(konfig)
    konfig.MESH_FILENAME='mesh.su2'

    return konfig
#Execute CFD computation
def runCFD(config):

    state = SU2.io.State()
    info = SU2.run.CFD(config)
    state.update(info)

    return state

def gen_parent():
    return np.ndarray.tolist((maxi-mini)*np.random.random_sample(L)+mini)

def gen_population():
    P=[]
    for i in range(N):
        P.append(gen_parent())
    return P

def parallel_evaluate(config, p):
	config=deform(config, p)
	state=SU2.io.State()
	config.OBJECTIVE_FUNCTION='DRAG'
	drag=SU2.eval.func('DRAG',config,state)
	obj=drag*-0.001
	return obj

def eval_pop(config, P):
	fit=[]
	#fit= Parallel(n_jobs=8)(delayed(parallel_evaluate(config,p)) for p in P)
	for p in P:
		fit.append(parallel_evaluate(config, p))
	return fit
    
def eval_alternate(P):
    fit=[]
    for p in P:
        obj=sum(p)/L
        fit.append(obj)
    return fit

def selection_tournament(fit,P):
    P1=[]
    for i in range(N):
        i_a=np.random.randint(0,N-1)
        i_b=np.random.randint(0,N-1)
        if(fit[i_a]>=fit[i_b]):
            P1.append(P[i_a])
        else:
            P1.append(P[i_b])
    return P1

def selection_prob(fit,P):
    P1=[]
    fit=np.asarray(fit)+5
    probs=fit/np.sum(fit)
    
    for n in range(N):    
        p=np.random.random()
        i=1
        while(p>=np.sum(probs[:i])):
            i+=1
        P1.append(P[i-1])
    return P1    
    
    

def combination_single_cross(p_c,P):
    P1=[]
    for i in range(N):
        p=np.random.random()
        if(p>p_c):
            i_a=np.random.randint(0,N-1)
            i_b=np.random.randint(0,N-1)
            i_c=np.random.randint(0,L-1)
            son=(P[i_a][:i_c])+(P[i_b][i_c:])
            P1.append(son)
        else:
            P1.append(P[i])
    return P1        

def mutation(p_m,P):
    P1=[]
    for i in range(N):
        p=np.random.random()
        if(p>p_m):
            i_m=np.random.randint(0,L-1)
            mut=P[i][:i_m]+[(maxi-mini)*np.random.random()+mini]+P[i][i_m+1:]
            P1.append(mut)
        else:
            P1.append(P[i])

    return P1
      





# this is only accessed if running from command prompt
if __name__ == '__main__':
    main()
