import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from random import shuffle
import collections

def shufGen(n,leng):
    '''
    Make a shuffled list of length 'leng' which has a correlation coefficent below 0.1 for every section of 'n' consecutive numbers.

    Parameters
    ----------
    n : int
        The number of values between 5-95% that will be used to create the list and the number over which to calculate the correlation coefficent.
        Generally this number is the same number of gradient slices used to fit the NMR diffusion data.
    leng : int
        The desired length of the shuffled list.

    Returns
    -------
    optGs : DataFrame
        Shuffled list (with correlation coefficent of each section of n below 0.1, and no number repeated within 8 consecutive numbers).
    corrs : DataFrame
        Correlation coefficients of each consecutive n numbers in optGs.

    '''
    phase = np.linspace(1,n,n) # define integer list of length n
    Mono = np.linspace(5,95,n) # define list of equally spaced values between 5-95% of length n
    start = Mono.copy()
    # shuffled monotonic list until it has a sufficiently low correlation
    while np.abs(pearsonr(phase,start)[0]) > 0.01: 
        shuffle(start)
    # grow list, one element at a time, ensure a sufficeintly low correlation with each addition
    def shufAdd(optGs,corrs,leng):
        count = 0
        adds = 0
        while len(optGs) < leng:
            count += 1
            shuf = Mono.copy()
            shuffle(shuf) # shuffle list every time to avoid favoring early numbers in list
            for g in shuf:
                # only add value if it does not appear in the previous n/2 values of list
                if g not in optGs[-int(n/2):]:
                    optGs.append(g) # add element
                    corr,p = pearsonr(phase,optGs[-n:]) # check correlation of the last n values (including the one just added)
                    # keep the element if correlation is low enough
                    if np.abs(corr) < 0.1:
                        corrs.append(corr)
                        adds += 1
                    # remove the element if correlation is too high
                    else:
                        del optGs[-1]
            if count > adds:
                break # stop the loop if no more elements are being added (will continue looping forever if not)
        return optGs, corrs
    optGs = start.tolist() # begin shuffled list with a sufficiently low correlation
    count = 0
    lenGs = [] # collect the lengths of optGs over each iteration to return the longest if the desired length is not reached 
    while len(optGs) < leng: # reset the shufAdd loop until you reach the desired length
        count += 1
        print('Iteration#',count)
        optGs, corrs = shufAdd(start.tolist(),[pearsonr(phase,start.tolist())[0]],leng)
        lenGs.append(len(optGs))
        if len(optGs) == max(lenGs): # collect the longest result over each iteration (to be used if the desired length is not reached)
            bestGs = optGs
            bestcorrs = corrs
        if count == 1000: # give up after 1000 iterations, there may be result it hasn't found due to the random nature of the loops (if you press it again it may find one immediately )
            optGs = bestGs # return the longest if the desired length is not reached after 1000 iterations
            corrs = bestcorrs
            print('Try shorter list')
            break
    print('List of length %(n)1.0f reached after %(c)1.0f iterations' %{'n':len(optGs),'c':count})
    return pd.DataFrame(optGs,columns=['Value / %']), pd.DataFrame(corrs,columns=['Correlation'])
    
#o,c=shufGen(16,500)
#o.to_csv('Shuffled list.csv')

#%%
def altshufGen(n,leng):
    '''
    Make a shuffled list of length 'leng' which has a correlation coefficent below 0.01 for every section of 'n' consecutive numbers.
    The list will (mainly) alternate between numbers in the first half and second half of a monotonic list of length n between 5-95.
    
    Parameters
    ----------
    n : int
        The number of values between 5-95% that will be used to create the list and the number over which to calculate the correlation coefficent.
        Generally this number is the same number of gradient slices used to fit the NMR diffusion data.
    leng : int
        The desired length of the shuffled list.

    Returns
    -------
    optGs : DataFrame
        Shuffled list (with correlation coefficent of each section of n below 0.1, and no number repeated within 8 consecutive numbers).
    corrs : DataFrame
        Correlation coefficients of each consecutive n numbers in optGs.

    '''
    phase = np.linspace(1,n,n) # define integer list of length n
    Mono = np.linspace(5,95,n) # define list of equally spaced values between 5-95% of length n
    small = Mono[:int(n/2)]
    big = Mono[int(n/2):]
    # create monotonic list to start from (just 0,1,2...n, numbers will be replaced just need it to be the right size)
    start=np.linspace(0,n,n) 
    # loop until a sufficiently small correlation coeff is achieved
    while np.abs(pearsonr(phase,start)[0]) > 0.01:
        shuffle(small) # shuffle first half of the number
        start[::2]=small # replace every other number in start (starting from 0) with new shuffled list of small numbers
        shuffle(big) # shuffle second half of the number
        start[1::2]=big # replace every other number in start (starting from 1) with new shuffled list of big numbers
    # function to grow list, one element at a time, ensure a sufficeintly low correlation with each addition
    def altshufAdd(optGs,corrs,leng):
        count = 0 # counting number of loops and additions (to list) means you can stop and restart when it gets stcuk
        adds = 0
        while len(optGs) < leng:
            count += 1 # +1 loop
            shuffle(small) # shuffle lists every time to avoid favoring early numbers
            shuffle(big)
            for shuf in [small,big]: # try to ensure every other number in optGs is small,big,small,big
                for g in shuf:
                    # only add numbers that do not appear in the previous n/2 entries
                    if g not in optGs[-int(n/2):]:
                        optGs.append(g)
                        corr,p = pearsonr(phase,optGs[-n:]) # check correlation of the last n values (including the one just added)
                        if np.abs(corr) < 0.05:
                            corrs.append(corr)
                            adds += 1 # +1 addition
                            break
                        else:
                            del optGs[-1]
            if count > adds:
                break # stop the loop if no more elements are being added (will continue looping forever if not)
        return optGs, corrs
    optGs = start.tolist() # begin shuffled list with a sufficiently low correlation
    count = 0
    lenGs = [] # collect the lengths of optGs over each iteration to return the longest if the desired length is not reached 
    while len(optGs) < leng: # reset the altshufAdd loop until you reach the desired length
        count += 1
        print('Iteration#',count)
        optGs, corrs = altshufAdd(start.tolist(),[pearsonr(phase,start.tolist())[0]],leng)
        lenGs.append(len(optGs))
        if len(optGs) == max(lenGs): # collect the longest result over each iteration (to be used if the desired length is not reached)
            bestGs = optGs
            bestcorrs = corrs
        if count == 10: # give up after 1000 iterations, there may be result it hasn't found due to the random nature of the loops (if you press it again it may find one immediately )
            optGs = bestGs # return the longest if the desired length is not reached after 1000 iterations
            corrs = bestcorrs
            print('Try shorter list')
            break
    print('List of length %(n)1.0f reached after %(c)1.0f iterations' %{'n':len(optGs),'c':count})
    return pd.DataFrame(optGs,columns=['Value / %']), pd.DataFrame(corrs,columns=['Correlation'])

#o,c=altshufGen(32,512)
#o.to_csv('Shuffled list.csv')

