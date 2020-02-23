import csv
import matplotlib.pyplot as plt
import numpy as np
from Sidefunc import *
from DataReader import *
    
def data_hist(bb, nobb, name):                  #Plot an histogram for bbar particles and no bbar particles
    plt.hist(nobb, bins=50, label='nobb', edgecolor='r', alpha=0.5)
    plt.hist(bb,bins=50 ,label='bb', edgecolor='k', alpha=0.7)
    #plt.yscale('log', nonposy='clip')          #for y scale in log
    plt.legend(loc='best', fontsize='x-large')
    plt.title('bb and nobb for ' + name + ' from XX to XX')
    plt.show

def main(data, name):                           #Put in "data" the return of csv2dict function from DataReader.py.
    A,B = str2tuple(data, name)                 #Plot data from a chosen column.
    bb, nobb = sort(A,B)
    data_hist(bb, nobb, name)
