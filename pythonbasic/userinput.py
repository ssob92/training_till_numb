# inout is built-in function
# expression is an instruction that combines values and operators and always evaluates doen to a single value
# input always returns string

to_units = 24 * 60 * 60
units_name = "seconds"

def day_to_unit(days, custome_message):
    print(f"{days} days are {days * to_units} {units_name}")
    print("All good")
    print(custome_message)



days = input("Enter the number of days:\n")
   
 
day_to_unit(int(days), "youre in 40%")
try:    
    day_to_unit()
except TypeError:
    print("No value given")