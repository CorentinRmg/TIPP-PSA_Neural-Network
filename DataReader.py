"""
Here are the functions used to read data from csv file (and other functions related
 to the manipulation of those data) in order to use it to train neural networks.

@author: Corentin Roumegou / Sacha Cormenier
"""

import csv
import numpy as np

def csv2dict(file: str, newline_symbol:str='\n', delimiter_symbol:str=',')-> dict:
    """
    Takes a csv file as input and returns a dictionary
    with lists of data for each column whose names are the keys
    """
    #initializes an empty dictionary
    data={}
    
    #Creates one key per column of the csv file
    with open(file, newline=newline_symbol) as csvfile:
         filereader = csv.reader(csvfile, delimiter=',')
         names_list=[]
         for name_line in list(filereader)[0]:
             #creates a new empty list with each column name/key
             data[name_line]=[]
             #saves the names in a list for the following
             names_list.append(name_line)
    
    #fills the key's values with corresponding csv columns' values
    with open(file, newline='\n') as csvfile:
         filereader = csv.reader(csvfile, delimiter=delimiter_symbol)
         num_line=0
         for row in filereader:
             if num_line!=0:    #the first line corresponds to the names
                 for num_column in range(len(names_list)):
                     data[names_list[num_column]].append(row[num_column])
             num_line+=1
    
    #Returns the dictionary with the names
    return data



def norm(vect):
    '''
    Returns the normalized vector
    '''
    return vect/np.mean(vect)