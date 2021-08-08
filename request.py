# -*- coding: utf-8 -*-

import requests
import json
ip_address = "13.235.254.119"
port = "5000"
data = [[5.1, 3.5, 1.5, 0.4]]

url = 'http://{0}:{1}/predict/'.format(ip_address, port)

json_data = json.dumps(data)
header = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
response = requests.post(url, data = json_data, headers = header)
print(response, response.text)
