import sys
import os
import requests

import http.client, urllib

requests.packages.urllib3.util.connection.HAS_IPV6 = False

USER = "u957ugfwotcdsngjumoqyruukhfkr2"
API = "ausacuqa5urtmhytuc8of2tak72cwa"

r = requests.get("https://www.google.com/")
print(r)

r = requests.post("https://api.pushover.net/1/messages.json", data={"token":API,"user":USER,"message":"Got Image?"})