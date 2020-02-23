import numpy as np

def str2tuple(data, colonne):
    return list(map(float,data[colonne])), list(map(float,data['isBB']))  #Take data from a chosen column of the file. Convert it in floats.

def sort(data, result):
    bb=[]
    nobb=[]
    for i in range (0,len(data)):               #Fill a tuple with bbar particles, and another with no bbar particles
        if result[i]==1:
            bb.append(data[i])
        else :
            nobb.append(data[i])
    return (bb, nobb)
