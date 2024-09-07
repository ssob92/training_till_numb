
# modules logically organize your python code
# 

#import helper as hp
from helper import *
import logging

to_units = 24 * 60 * 60
units_name = "seconds"
run_next = 1

logger = logging.getLogger("MAIN")
logger.error("Error happpened in the app")


while run_next == 1:
    days = input("Enter the number of days and conversion unit:\n")
    days_and_unit = days.split(":")
    daysunitdict = {"days": days_and_unit[0], "unit": days_and_unit[1]}
    validate_and_execute(daysunitdict)
    print(userinput)
    to_next_run = input("Wish to continue? 1= Yes, 0= No \n")
    run_next = int(to_next_run)