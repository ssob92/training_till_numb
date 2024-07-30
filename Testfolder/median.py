# Python3 code to demonstrate working of 
# Median of list 
# Using loop + "~" operator 

# initializing list 
test_list = [4, 5, 8, 9, 10, 17] 

# printing list 
print("The original list : " + str(test_list)) 

# Median of list 
# Using loop + "~" operator 
test_list.sort() 
mid = len(test_list) // 2
newmid = ~mid
test = test_list[~mid]
res = (test_list[mid] + test_list[~mid]) / 2

# Printing result 
print("Median of list is : " + str(res)) 
