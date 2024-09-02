import requests as rq

responnse = rq.get("https://gitlab.com/api/v4/users/nanuchi/projects")
#print(responnse.text)
print(responnse.json()[0])

my_projects = responnse.json()

for element in my_projects:
    print(element["name"])
    print(element["http_url_to_repo"])
    print("\n")
    