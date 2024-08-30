
max_fibo_num = 15
min_fibo_num = 0
start_fibo_num = 1
next_fibo_num = start_fibo_num
counter_fibo_num = 1

while counter_fibo_num <= max_fibo_num:
    try:
        print("[%s] Next fibonacci number: " %counter_fibo_num, next_fibo_num)
        counter_fibo_num += 1
        min_fibo_num,start_fibo_num = start_fibo_num,next_fibo_num
        next_fibo_num = min_fibo_num + start_fibo_num
    
    except:
        print("Something went wrong")