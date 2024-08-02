    # Slicing values
    min_values = arr[:-1]
    max_values = arr[1:]
    
    # Sum of the new list
    min_result = sum(min_values)
    max_result = sum(max_values)
    
    # Print the values out
    #print(min_result, max_result)
    print(min_values,max_values)
    #print(max_result)
    
     x = 0
    new_arr = []
    
    while (x <= len(arr)):
        del arr[x]
        new_sum = sum(arr)
        new_arr.append(new_sum)
        x+=1
        




#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'miniMaxSum' function below.
#
# The function accepts INTEGER_ARRAY arr as parameter.
#

def miniMaxSum(arr):
    # Write your code here
    
    #variable init for x and new empty list
    x = 0
    new_arr = []
    
    # create new set of list , by adding 4 out of 5 values from values and repeating the process till all the values was removed 
    while (x <= len(arr)-1):
        new_sum = sum(arr) - arr[x]
        new_arr.append(new_sum)
        x+=1
    
    
    
    # Print the values out
    #print(new_arr, arr)
    print(min(new_arr), max(new_arr))
    #print(max_result)
    
    
if __name__ == '__main__':

    arr = list(map(int, input().rstrip().split()))

    miniMaxSum(arr)
    

def timeConversion(s):
    # Write your code here
    
    # check if the last 2 elements is AM and first 2 element is 12
    if s[-2:] == "AM" and s[:2] == "12":
        return "00" + s[2:-2]
    
    # check if the last 2 element is AM
    elif s[-2:] == "AM":
        return s[:-2]
    
    # check if last 2 element is PM and first 2 item is 12    
    elif s[-2:] == "PM" and s[:2] == "12":
        return s[:-2]
    
    else:
        # return value with with first 2 elemts converted to int , added 12 and join back with remaining elemts
        # removing PM
        return str(int(s[:2]) + 12) + s[2:-2]
        
        

Find unique values
to_set = set(a)
new_unique = (list(to_set))
    
    for x in new_unique:
        return x
        
        
 def lonelyinteger(a):
    # Write your code here
    
    #new_value = []
    
    for ori in a:
        x = op.countOf(a, ori)
        
        if x > 1:
            continue
        else:
            return ori
            break