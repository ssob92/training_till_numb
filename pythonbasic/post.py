class Post:
    
    def __init__(self, message, author):
        self.message = message
        self.author = author
    
    def change_message(self, newmessage):
        self.message = newmessage
    
    def assign_new_author(self, new_author):
        self.author = new_author
    
    def get_all_details(self):
        print(f"Author: {self.author}, Message: {self.message}")