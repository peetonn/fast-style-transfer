# Semantic Image Segmentation Web Service

This service is built atop [TensorFlow CNN for fast style transfer](https://github.com/lengstrom/fast-style-transfer) implementation. For detailed information, please refer to the README of this repo.

## Dependencies

* install [TensorFlow](https://www.tensorflow.org/install/install_linux) (as of 3/17/2018 production, TensorFlow 1.5.0 was used)
* install requirements:

```
pip install pillow scipy numpy tornado
```

* in repo directory, create folders:

```
mkdir upload results
```

* download pre-trained models from [here](https://drive.google.com/drive/folders/0B9jhaT37ydSyRk9UX0wwX3BpMzQ) (author's link) or ask *peter [at] remap [dot] ucla [dot] edu* for more models (10 more models trained by us); **save models into *ce-models* folder, each model into a separate subfolder, named accordingly to each model name**.

    I.e. you should have folder structure, as follows:
    `<repo-directory>`
     * `ce-models`
       * `style-model1`
         * `<model file or files>`
       * `style-model1`
         * `<model file or files>`
       * `...`


## Configuration

* Since I didn't have time to figure out how to get IP address from within the running tornado service, modify [tornado/run.py](tonrado/run.py#L43) to your current public IP of the machine where this service will run.
* Unfortunately, once initialized, neural network can run only on one resolution. Thus, you need to decide which image resolution you will require and configure it accordingly in [tornado/run.py](tonrado/run.py#L266) file by choosing appropriate sample image. You can also add your own sample image for custom resoution.

## Run

Start service:

```
python tornado/run.py
```

## Check

Go to `http://<ip-address>:8889` - you should be able to see upload page where you can upload an image for segmentation. Once uploaded, service will return URL which you'll need to poll for the result (or it will return **code 423** if service is busy processing previous upload).

## API

* `/result/<FILE-ID>` - returns 404 if result was not processed yet or not found or returns segmented image (PNG);
* `/info` - returns JSON dictionary with keys:
    * `models` - an array of currently supported style models;
    * `res` - supported image resolution (dictionary with keys `w` and `h`);
* `/status` - returns **ok** if service runs normally.

## Handy 

* `curl` command for uploading images:

```
curl -F "file=@<path to file>" http://<ip-address>:8889/upload?style=lion-1
```

* python code for uploading images, checking the result and saving it into a file:

```
import requests
from time import sleep

port = 8889
ipaddress = <ip-address>
hostUrl = "http://"+ipaddress+":"+str(port)

uploadUrl = hostUrl+"/upload"

def upload(fname):
	imageFile = {'file': open(fname, 'rb')}
	style = 'udnie.ckpt'
	response = requests.post(uploadUrl, files=imageFile, params={'style':style})
	if response.status_code == 200:
		print("Upload successful.")
		statusCode = 0
		it = 0
		maxIter = 10
		maxWait = 2000
		while statusCode != 200 and it < maxIter:
			print("Fetching result "+str(it+1)+"/"+str(maxIter)+"...")
			r = requests.get(response.text, stream=True)
			statusCode = r.status_code
			it += 1
			if statusCode != 200:
				sleep(float(maxWait)/float(maxIter)/1000)

		if statusCode == 200:
			fname = "./result.png"
			with open(fname, 'wb') as f:
				for chunk in r:
					f.write(chunk)
			print("saved result at "+fname)
		else:
			print("time out receiving result from the server")

def main():
	upload("./sample420x236.jpg")

if __name__ == "__main__":
	main()
```
