from __future__ import print_function
import os, sys, signal

# adding import path for the directory above this sctip (for deeplab modules)
myPath = os.path.dirname(sys.argv[0])
rootPath = os.path.join(myPath,'..')
uploadPath =  os.path.join(rootPath, "upload")
resultsPath = os.path.join(rootPath, "results")
weightsModelPath = os.path.join(rootPath, "deeplab_resnet.ckpt")

sys.path.append(rootPath)

import tornado.httpserver, tornado.ioloop, tornado.options, tornado.web, os.path, random, string
import uuid
from tornado.options import define, options
from Queue import Queue
from threading import Thread
from datetime import datetime

import time
import datetime
from PIL import Image
import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, dense_crf, prepare_label

SAVE_DIR = './output/'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
GPU=1
###

port = 8888
ipaddress = "131.179.142.7"
hostUrl = "http://"+ipaddress+":"+str(port)
define("port", default=port, help="run on the given port", type=int)

quit = False
requestQueue = Queue()

def timestampMs():
    return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", IndexHandler),
            (r"/upload", UploadHandler),
            (r"/result/(.*)", tornado.web.StaticFileHandler, {"path" : "./results"})
        ]
        tornado.web.Application.__init__(self, handlers)
        
class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("upload_form.html")
        
class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        print("New upload request "+str(self.request))

        fileData = self.request.files['file'][0]
        original_fname = fileData['filename']
        extension = os.path.splitext(original_fname)[1]
        fileID = str(uuid.uuid4())
        fname = os.path.join(uploadPath, fileID)
        imageFile = open(fname, 'w')
        imageFile.write(fileData['body'])

        requestQueue.put(fileID)
        print("Submitted request " + fileID + " for segmentation processing");

        self.finish(hostUrl+"/result/"+fileID+".png")

### DEEPLAB STUFF BELOW
def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def deeplabProcessing(gpuId):
    """Create the model and start the evaluation process."""
    print("Starting worker on GPU "+str(gpuId) + "...")

    def printWorker(msg):
        print(str(timestampMs())+" [gpu-worker-"+ str(gpuId)+"] " + msg)

    printWorker("Waiting for segmentation requests...")
    initialized = False
    while (not quit):
        fileId = requestQueue.get() # will block
        if fileId == "quit"+str(gpuId):
            printWorker("Received quit command")
            break

        printWorker("Received request for DL segmentaiton: "+fileId)

        t1 = datetime.datetime.now()

        imgPath = os.path.join(uploadPath, fileId)
        # Prepare image.
        imgRGB = tf.image.decode_jpeg(tf.read_file(imgPath), channels=3)
        # Convert RGB to BGR.
        img_r, img_g, img_b = tf.split(imgRGB, 3, axis=2)
        imgBGR = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
        # Extract mean.
        imgBGR -= IMG_MEAN 

        printWorker("Will create network")
        # Create network.
        net = DeepLabResNetModel({'data': tf.expand_dims(imgBGR, dim=0)}, is_training=False)
        tf.get_variable_scope().reuse_variables()
    
        printWorker("Network created")

        # Which variables to load.
        restore_var = tf.global_variables()

        # Predictions.
        raw_output = net.layers['fc1_voc12']
        raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(imgBGR)[0:2,])
    
        printWorker("Predictions")

        # CRF.
        raw_output_up = tf.nn.softmax(raw_output_up)
        raw_output_up = tf.py_func(dense_crf, [raw_output_up, tf.expand_dims(imgRGB, dim=0)], tf.float32)
    
        printWorker("CRF")

        raw_output_up = tf.argmax(raw_output_up, dimension=3)
        pred = tf.expand_dims(raw_output_up, dim=3)

        if not initialized:
            printWorker("Setup tf session")
            # Set up TF session and initialize variables. 
            config = tf.ConfigProto(device_count = {'GPU': gpuId})
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
    
            sess.run(init)
            printWorker("TF session initialized")

            # Load weights.
            loader = tf.train.Saver(var_list=restore_var)
            load(loader, sess, weightsModelPath)

            initialized = True
    
        # Perform inference.
        preds = sess.run(pred)
    
        msk = decode_labels(preds)
        im = Image.fromarray(msk[0])
        maskPath = os.path.join(resultsPath, fileId)+".png"
        im.save(maskPath)
        
        originalFile = os.path.join(uploadPath,fileId)
        os.remove(originalFile)

        t2 = datetime.datetime.now()
        printWorker("Processing took "+str((t2-t1).microseconds)+"usec. Result is at "+maskPath)

def signal_handler(signum, frame):
    global is_closing
    print('Received stop signal, exiting...')
    tornado.ioloop.IOLoop.instance().stop()
    quit = True

def main():
    signal.signal(signal.SIGINT, signal_handler)

    # TODO: this can be expanded to utilize more than one GPU
    nGpus = 1
    workers = []
    for i in range(0,nGpus):
        worker = Thread(target=deeplabProcessing, args=(i,))
        worker.start()
        workers.append(worker)

    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
    
    print("Will terminate GPU workers...")

    for i in range(0,len(workers)):
        requestQueue.put("quit"+str(i))

if __name__ == "__main__":
    main()
