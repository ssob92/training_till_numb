

def day_to_unit(days, conversionunit):
    if conversionunit == "hours":
        return f"{days} days are {days * 24} {conversionunit}\n"
    elif conversionunit == "minutes":
        return f"{days} days are {days * 24 * 60} {conversionunit}\n"
    else:
        return "unsupported unit"

        
def validate_and_execute(daysunitdicti):
    
    try:
            days_int = int(daysunitdicti["days"])
            if days_int > 0:
                calculated_values = day_to_unit(days_int,daysunitdicti["unit"])
                print(calculated_values)
            elif days_int == 0:
                print("Zero youuuu")       
    except ValueError:
        print("Dont ruin my programme")
        

userinput = "this values are from module"