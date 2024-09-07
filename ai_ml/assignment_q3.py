import datetime as dt

date_time_now = dt.datetime.now()
username = input("Please enter your nickname: ")
birthdate_day = input("Please enter your birth date(Day): ")
birthdate_month = input("Please enter your birth date(Month): ")
birthdate_year = input("Please enter your birth date(Year): ")

convert_to_datetime = dt.datetime(int(birthdate_year),int(birthdate_month),int(birthdate_day))
username_age = (date_time_now-convert_to_datetime)/365

print(f"Welcome {username}")
print(f"Today's date/time is {date_time_now}")
print(f"Your birthdate is {convert_to_datetime}")
print(f"Your age is {username_age}")
