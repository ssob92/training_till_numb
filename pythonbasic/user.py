class User:
    
    def __init__(self, email, name, password, current_job_title):        
        self.email = email
        self.name = name
        self.password = password
        self.current_job_title = current_job_title
    
    def change_password(self, newpassword):
        self.password = newpassword
    
    
    def change_job_title(self, new_job_title):
        self.current_job_title = new_job_title
        
    def get_user_info(self):
        print(f"Hi , I'm {self.name}, working as {self.current_job_title}. You can contact me at {self.email}")
        
