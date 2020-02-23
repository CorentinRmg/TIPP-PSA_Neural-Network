import csv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
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

def plot_dot(bb1,bb2,nobb1,nobb2, name1, name2):
    plt.plot(nobb1,nobb2,'bo', markersize=2.2)
    plt.plot(bb1,bb2,'ro', markersize=2.2)
    plt.xlabel(name1)
    plt.yscale('log', nonposy='clip')
    plt.xscale('log')
    plt.ylabel(name2)
    plt.title('bb and nobb '+name1+' & '+name2+ ' from 50 000 to 69 999')
    plt.show()

def main(data, name1, name2):                           #Put in data the return of csv2dict function from DataReader.py
    A,B = str2tuple(data, name1)
    C,D = str2tuple(data, name2)
    bb1, nobb1 = sort(A,B)
    bb2, nobb2 = sort(C,D)
    plot_dots3D(bb1,bb2,nobb1,nobb2,name1,name2)
