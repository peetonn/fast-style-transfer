import requests
from time import sleep

port = 8888
ipaddress = "127.0.0.1" # "131.179.142.7"
hostUrl = "http://"+ipaddress+":"+str(port)

uploadUrl = hostUrl+"/upload"

def upload(fname):
	imageFile = {'file': open(fname, 'rb')}
	response = requests.post(uploadUrl, files=imageFile)
	if response.status_code == 200:
		print("Upload successful.")
		statusCode = 0
		it = 0
		maxIter = 10
		maxWait = 2000
		while statusCode != 200 and it < maxIter:
			print("Fetching result "+str(it+1)+"/"+str(maxIter)+"...")

			r = requests.get("http://127.0.0.1:8888/result/fabd1020-a705-463d-b386-c60048d20c1a", stream=True)
			statusCode = r.status_code
			it += 1
			if statusCode != 200:
				sleep(float(maxWait)/float(maxIter))

		if statusCode == 200:
			fname = "./image.png"
			with open(fname, 'wb') as f:
				for chunk in r:
					f.write(chunk)
			print("saved image at "+fname)
		else:
			print("time out receiving result from the server")

def main():
	upload("/Users/peetonn/Downloads/img12.png")

if __name__ == "__main__":
	main()



