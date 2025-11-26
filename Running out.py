from math import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

fancy=True ; np.random.seed(42)

def seven_segment(pattern):
    global fancy;fancy = True
    left,lower,right = \
        ('▕','__','▏') if fancy else \
        ('|','__','|')
    def vert(d1,d2,d3):
        print(
        (left  if d1 else " ")+
        (lower if d3 else "  ")+
        (right if d2 else " "))
    bits = [(ai==1) for ai in pattern]
    print(" "+lower+" " if bits[0] else "   ")
    vert(*bits[1:4])
    vert(*bits[4:7])
    number=0
    for i in range(0,4):
        if bits[7+i]:
            number+=pow(2,i)
    print('%X'%int(number))
    


hexdigits = np.int8([\
    [1,1,1,0,1,1,1, 0,0,0,0],
    [0,0,1,0,0,1,0, 1,0,0,0],
    [1,0,1,1,1,0,1, 0,1,0,0],
    [1,0,1,1,0,1,1, 1,1,0,0],
    [0,1,1,1,0,1,0, 0,0,1,0],
    [1,1,0,1,0,1,1, 1,0,1,0],
    [1,1,0,1,1,1,1, 0,1,1,0],
    [1,0,1,0,0,1,0, 1,1,1,0],
    [1,1,1,1,1,1,1, 0,0,0,1],
    [1,1,1,1,0,1,1, 1,0,0,1],
    [1,1,1,1,1,1,0, 0,1,0,1],
    [0,1,0,1,1,1,1, 1,1,0,1],
    [1,1,0,0,1,0,1, 0,0,1,1],
    [0,0,1,1,1,1,1, 1,0,1,1],
    [1,1,0,1,1,0,1, 0,1,1,1],
    [1,1,0,1,1,0,0, 1,1,1,1]])*2-1

zero, one, two, three, four, five, six, seven, eight, nine, hexA, hexB, hexC, hexD, hexE, hexF = hexdigits
all_pats = np.array([zero,one, two, three, four, five, six, seven, eight, nine, hexA, hexB, hexC, hexD, hexE, hexF]);

test1 = np.int8([1,-1,1,1,-1,1,1,-1,-1,-1,-1]) #to 3
test2 = np.int8([1,1,1,1,1,1,1,-1,-1,-1,-1]) # to 6
test3 = np.int8([-1,1,1,-1,-1,1,-1,-1,-1,-1,-1]) # to 4
test4 = np.int8([1,1,-1,-1,1,-1,1,-1,-1,-1,1]) #  to c

def enc_pats():
    output = []
    for pat in all_pats:
        output.append(enc(pat))
    return np.array(output)

def compare(x_in,pats):
    sims = []
    for i in pats:
        sims.append(round(esim(x_in,i),5))
    print(sims)

def converge_energy(s,W,maxiter=20,printend=False,printenergy=False):
    previous_energy = E(s,W)
    for i in range(maxiter):
        if printenergy: print('iteration %2d: E = %6.2f'%(i,previous_energy))
        x = W@s
        s = Kwins(W@s)
        compare(s,X)
        energy = E(s,W)
        if energy>=previous_energy: break
        previous_energy = energy
    return s

def converge_states(s,W,maxiter=20,printend=False,printenergy=False):
    ps = None # detect 2-cycles
    for i in range(maxiter):
        if printenergy: 
            print('iteration %2d: E = %6.2f'%(i,E(s,W)))
        s_ = Kwins(W@s)
        if np.all(s==s_): break
        if ps is not None and np.all(ps==s_): 
            print("cycle detected")
            break
        ps = s; s = s_
    if printend:
        seven_segment(s)
        print(E(s,W))
        print()
    return s

def similarity_MAT(pats):
    simmat = [] 
    for i in pats:
        simvec = []
        for j in pats:
            simvec.append(esim(i,j))
        simmat.append(np.array(simvec))
    return np.array(simmat)

N_v = 11; N_h = 200 ; spars = 0.02

W_dg, _ = np.linalg.qr(np.random.normal(size=(N_h,N_v)));   # gen random matrix and mean subtract to remove bias
# W_dg = np.random.normal(size=(N_h,N_v)); W_dg = W_dg - np.mean(W_dg);

#lambda functions
Kwins =  lambda x: (x >= np.partition(x, -int(N_h*spars))[-int(N_h*spars)]).astype(float)*2 - 1
enc = lambda x: Kwins(W_dg @ x.T); sign = lambda x: 2*(x>0) - 1 ; nm = lambda x :np.linalg.norm(x)
E = lambda x,W:-0.5*x.T@W@x; sign = lambda x: 2*(x>0) - 1; esim = lambda a, b: ((np.dot(a,b)/(nm(a)*nm(b))) +1)/2
reformat = lambda x: (x - np.min(x))/(np.max(x)-np.min(x)); ham =lambda a,b: np.sum((a!=b))

if __name__ =='__main__':
    X = enc_pats(); W = (X + 1 - 2*spars).T @ (X + 1 - 2*spars); np.fill_diagonal(W,0); W = W - np.mean(W)
    
    def run_test(test,actual):
        tf_test = enc(test)
        compare(tf_test,X)
        out1 = converge_states(tf_test,W,printenergy=True)
        compare(out1,X)
        print(np.all(out1==enc(actual)))
        print(ham(tf_test,enc(actual)))
        print('\n-------------------------------------\n')
    
    # for i in range(len(X)):
    #     run_test(all_pats[i],all_pats[i])
        
    run_test(test1,three)
    run_test(test2,six)
    run_test(test3,four)
    run_test(test4,hexC)   
    
    
    
    
    
    
    
    
    unencsim = reformat(similarity_MAT(all_pats)); encsim = reformat(similarity_MAT(X))

    fig, (ax1,ax2) = plt.subplots(1,2); cbar_ax = fig.add_axes([.95, 0.2, 0.03, 0.6])
    ax1.imshow(unencsim,vmin=0, vmax=1); im = ax2.imshow(encsim,vmin=0, vmax=1)
    ax1.set_xticklabels([]);ax2.set_xticklabels([]);ax1.set_yticklabels([]);ax2.set_yticklabels([])
    ax1.tick_params(bottom=False, left=False);ax2.tick_params(bottom=False, left=False)
    fig.colorbar(im, cax=cbar_ax); fig.subplots_adjust(wspace = 0.3)
    ax1.set_title('Unencoded Patterns'); ax2.set_title("Sparse Encoded Patterns")









