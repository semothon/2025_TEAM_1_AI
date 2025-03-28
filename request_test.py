# import requests

# url = "http://127.0.0.1:5000/compare_images"
# files = {
#     "file1": open("kyungpo1.jpeg", "rb"), 
#     "file2": open("kyungpo1.jpeg", "rb"),
# }

# response = requests.post(url, files=files)

# print(response.json())
import requests

url = "http://127.0.0.1:5000/happiness"
files = {
    "file": open("winter2.jpg", "rb")
}

response = requests.post(url, files=files)

print(response.json()) 
