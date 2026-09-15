# 14.1.py
import urllib.request

response = urllib.request.urlopen("https://unsplash.it/1600/900?random")
_img = response.read()

with open("img_1600_900.jpg", "wb") as f:
    f.write(_img)
