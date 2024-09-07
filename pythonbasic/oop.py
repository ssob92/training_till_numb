from user import User
from post import Post


app_user_one = User("test@nn.com", "Bala Sambasevam", "test123", "DevOps Data Engineer")
app_user_one.get_user_info()

app_user_one.change_job_title("Data Scientist")
app_user_one.get_user_info()

app_user_two = User("test2@gmail.com", "Julia", "2021", "Call Agent")
app_user_two.get_user_info()

app_user_two.change_job_title("Gov Lawyer")
app_user_two.get_user_info()

author_one = Post("This is new", "JK Rowling")
author_two = Post("Whatever", "Jivy")

author_one.get_all_details()
author_two.get_all_details()

author_one.assign_new_author("Steven SPielberg")
author_two.change_message("I may have reached 45%")

print("After Changes")
author_one.get_all_details()
author_two.get_all_details()