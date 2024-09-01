
# function can return results data as value


to_units = 24 * 60 * 60
units_name = "seconds"

def day_to_unit(days, custome_message):
    status = f"{days} days are {days * to_units} {units_name}\n"
    return status,custome_message


days = input("Enter the number of days:\n")
message = input("Type in your message \n")
calculated_values = day_to_unit(int(days),message)
 

#whats_the_status = day_to_unit(45,"\n Yore still at 40%")
 
'''
day_to_unit(int(days), "youre in 40%")
try:    
    day_to_unit()
except TypeError:
    print("No value given")
'''