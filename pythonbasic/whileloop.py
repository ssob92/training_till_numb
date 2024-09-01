
# looping is to execute the logic multiple times or until condition is met

to_units = 24 * 60 * 60
units_name = "seconds"
run_next = 1

def day_to_unit(days, custome_message):
    
    status = f"{days} days are {days * to_units} {units_name}\n"
    return status,custome_message
        
def validate_and_execute():
    
    try:
            days_int = int(days)
            if days_int > 0:
                calculated_values = day_to_unit(days_int,message)
                print(calculated_values)
            elif days_int == 0:
                print("Zero youuuu")       
    except ValueError:
        print("Dont ruin my programme")


while run_next == 1:
    days = input("Enter the number of days:\n")
    message = input("Type in your message \n")
    validate_and_execute()
    to_next_run = input("Wish to continue? 1= Yes, 0= No \n")
    run_next = int(to_next_run)