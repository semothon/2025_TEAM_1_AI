import requests

url = "http://127.0.0.1:5000/compare_images"
files = {
    "url1": "https://demo-bucket-605134439665.s3.ap-northeast-2.amazonaws.com/wave.png", 
    "url2": "https://demo-bucket-605134439665.s3.ap-northeast-2.amazonaws.com/wave.png",
}

response = requests.post(url, json=files)

print(response.json())
# import requests

# url = "http://127.0.0.1:5000/happiness"
# files = {
#     "url": open("./test_images/test3.jpeg", "rb")
# }

# response = requests.post(url, files=files)

# print(response.json()) 
