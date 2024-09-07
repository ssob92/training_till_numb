

def add(num1,num2):
    return num1 + num2

def subtract(num1,num2):
    return num1 - num2

def multiply(num1,num2):
    return num1*num2

def divide(num1,num2):
    return num1/num2

num_latest = ""

print("Python Assignment Task 1")
print("========================================")

while True:
    
    if num_latest == "":
        try:
            num_1,opr,num_2 = input("Calculate 2 numbers:").split()
            
            if opr == "+":
                num_latest = add(int(num_1),int(num_2))
                print(f"{num_1} + {num_2} = {num_latest}")
                
            elif opr == "-":
                num_latest = subtract(int(num_1),int(num_2))
                print(f"{num_1} - {num_2} = {num_latest}")
                
            elif opr == "*":
                num_latest = multiply(int(num_1),int(num_2))
                print(f"{num_1} * {num_2} = {num_latest}")
            
            elif opr == "/":
                num_latest = divide(int(num_1),int(num_2))
                print(f"{num_1} + {num_2} = {num_latest}")
                
            else:
                print("Invalid input! Please try again...")
            
            opr_continue = input("Continue Calculation? Y/N")
            if opr_continue == "Y" or opr_continue == "y":
                print("Continue")
            
            else:
                print("No more calculation...")
                break
        
        except ZeroDivisionError:
            print("Cannot divide Zero")
    
    else:
        
        try:
            opr,num_2 = input("Calculate with the next new numbers:%s"%num_latest).split()
            
            if opr == "+":
                print(f"{num_latest} + {num_2} = {add(int(num_latest),int(num_2))}")
                num_latest = add(int(num_latest),int(num_2))
                
                
            elif opr == "-":
                print(f"{num_latest} - {num_2} = {subtract(int(num_latest),int(num_2))}")
                num_latest = subtract(int(num_latest),int(num_2))
                
                
            elif opr == "*":
                print(f"{num_latest} * {num_2} = {multiply(int(num_latest),int(num_2))}")
                num_latest = multiply(int(num_latest),int(num_2))
                
            
            elif opr == "/":
                print(f"{num_latest} + {num_2} = {divide(int(num_latest),int(num_2))}")
                num_latest = divide(int(num_latest),int(num_2))
                
            else:
                print("Invalid input! Please try again...")
            
            opr_continue = input("Continue Calculation? Y/N")
            if opr_continue == "Y" or opr_continue == "y":
                print("Continue")
            
            else:
                print("No more calculation...")
                break
        
        except ZeroDivisionError:
            print("Cannot divide Zero")


print("========================================")
quit()            
            