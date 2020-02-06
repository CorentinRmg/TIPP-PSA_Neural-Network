import csv
import matplotlib.pyplot as plt
import numpy as np
from DataReader import *

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

    
def data_hist(bb, nobb, name):                  #Plot an histogram for bbar particles and no bbar particles
    plt.hist(nobb, bins=50, label='nobb', edgecolor='r', alpha=0.5)
    plt.hist(bb,bins=50 ,label='bb', edgecolor='k', alpha=0.7)
    plt.legend(loc='best', fontsize='x-large')
    plt.title('bb and nobb for ' + name + ' from XX to XX')
    plt.show

def main(data, name):                           #Put in "data" the return of csv2dict function from DataReader.py.
    A,B = str2tuple(data, name)                 #Plot data from a chosen column.
    bb, nobb = sort(A,B)
    data_hist(bb, nobb, name)
