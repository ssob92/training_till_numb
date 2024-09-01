



to_units = 24 * 60 * 60
units_name = "seconds"

def day_to_unit(days, custome_message):
    if days > 0:
        status = f"{days} days are {days * to_units} {units_name}\n"
    elif days == 0:
        status = "Zero youuuuuu"
    else:
        status = "You've entered a negative value"
    
    return status,custome_message
        
def validate_and_execute():
    if days.isdigit():
        days_int = int(days)
        message = input("Type in your message \n")
        calculated_values = day_to_unit(days_int,message)
        print(calculated_values)
    else:
        print("Dont ruin my programme")


days = input("Enter the number of days:\n")
validate_and_execute()
