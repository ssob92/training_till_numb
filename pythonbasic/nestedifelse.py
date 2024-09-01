


to_units = 24 * 60 * 60
units_name = "seconds"

def day_to_unit(days, custome_message):
    
    status = f"{days} days are {days * to_units} {units_name}\n"
    return status,custome_message
        
def validate_and_execute():
    if days.isdigit():
        days_int = int(days)
        if days_int > 0:
            calculated_values = day_to_unit(days_int,message)
            print(calculated_values)
        elif days_int == 0:
            print("Zero youuuu")       
    else:
        print("Dont ruin my programme")


days = input("Enter the number of days:\n")
message = input("Type in your message \n")
validate_and_execute()