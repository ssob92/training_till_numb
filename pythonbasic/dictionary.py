

# dictionary can be accessed usingkey name


to_units = 24 * 60 * 60
units_name = "seconds"
run_next = 1

def day_to_unit(days, conversionunit):
    if conversionunit == "hours":
        return f"{days} days are {days * 24} {conversionunit}\n"
    elif conversionunit == "minutes":
        return f"{days} days are {days * 24 * 60} {conversionunit}\n"
    else:
        return "unsupported unit"

        
def validate_and_execute():
    
    try:
            days_int = int(daysunitdict["days"])
            if days_int > 0:
                calculated_values = day_to_unit(days_int,daysunitdict["unit"])
                print(calculated_values)
            elif days_int == 0:
                print("Zero youuuu")       
    except ValueError:
        print("Dont ruin my programme")


while run_next == 1:
    days = input("Enter the number of days and conversion unit:\n")
    days_and_unit = days.split(":")
    daysunitdict = {"days": days_and_unit[0], "unit": days_and_unit[1]}
    validate_and_execute()
    to_next_run = input("Wish to continue? 1= Yes, 0= No \n")
    run_next = int(to_next_run)