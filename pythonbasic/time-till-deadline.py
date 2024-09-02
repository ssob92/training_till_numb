import datetime as dt

userim = input("ENter your goal with a deadline separated by colon: \n")
userim_list = userim.split(":")

goal = userim_list[0]
deadline = userim_list[1]

# calculate how many days from now till deadline

today_date = dt.datetime.today()
dateline_date = dt.datetime.strptime(deadline, "%d.%m.%Y")
time_till = dateline_date - today_date
hours_till = int(time_till.total_seconds() / 60 / 60)



if int(time_till.days) == 0:
    print(f"You have {hours_till} hours to complete {goal} goal")
else:
    print(f"You have {time_till.days} days to complete {goal} goal")