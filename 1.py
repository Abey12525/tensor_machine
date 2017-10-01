import requests
s = requests.session()
login_data = dict(username='foss', password='foss@ajc')
s.post('http://192.168.0.1:8090/', data=login_data)
r = s.get('http://192.168.0.1:8090')
print (r.content)
