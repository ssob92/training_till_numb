
# functions is created to avoid repeated logics , defined using def, only runs when it called, informations can be passed as parameters


to_units = 24 * 60 * 60
units_name = "seconds"

def day_to_unit(days, custome_message):
    print(f"{days} days are {days * to_units} {units_name}")
    print("All good")
    print(custome_message)

    
 
day_to_unit(35, "youre in 40%")
try:    
    day_to_unit()
except TypeError:
    print("No value given")

'''   
print(f"20 days are {to_units} {units_name}")
print(f"35 days are {35 * to_units} {units_name}")
print(f"300 days are {50 * to_units} {units_name}")
'''