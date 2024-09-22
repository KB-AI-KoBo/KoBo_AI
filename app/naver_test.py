import os
import sys
import urllib.request
client_id = "VZqunGuAjPeTN1rIL10z"
client_secret="XQ6HSwgKi4"
encText = urllib.parse.quote("롯데마트")
url = 'https://openapi.naver.com/v1/search/news?query='+encText
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-id",client_id)
request.add_header("X-Naver-Client-Secret", client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)