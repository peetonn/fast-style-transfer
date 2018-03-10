from __future__ import print_function
import os, sys, signal
sys.path.insert(0, 'src')

# adding import path for the directory above this sctip (for deeplab modules)
myPath = os.path.dirname(sys.argv[0])
rootPath = os.path.join(myPath,'..')
uploadPath =  os.path.join(rootPath, "upload")
resultsPath = os.path.join(rootPath, "results")
modelsDir = os.path.join(rootPath, 'ce-models');

sys.path.append(rootPath)

import tornado.httpserver, tornado.ioloop, tornado.options, tornado.web, os.path, random, string
import uuid
from tornado.options import define, options
from Queue import Queue
from threading import Thread
from datetime import datetime
import re
import time
import datetime
from PIL import Image
import tensorflow as tf
import numpy as np
import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from collections import defaultdict
import time
import json
import subprocess
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer

BATCH_SIZE = 4
DEVICE = '/gpu:0'

port = 8889
ipaddress = "131.179.142.7"
hostUrl = "http://"+ipaddress+":"+str(port)
define("port", default=port, help="run on the given port", type=int)

allModels = []
sampleImg = None
sampleImgW = None
sampleImgH = None
quit = False
requestQueue = Queue()

#******************************************************************************
def timestampMs():
    return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", IndexHandler),
            (r"/upload", UploadHandler),
            (r"/result/(.*)", tornado.web.StaticFileHandler, {"path" : "./results"}),
            (r"/info", InfoHandler)
        ]
        tornado.web.Application.__init__(self, handlers)
        
class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        global allModels
        self.render("upload_form.html", imageWidth = sampleImgW, imageHeight = sampleImgH, models=allModels)
        
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

class InfoHandler(tornado.web.RequestHandler):
    def get(self):
        global allModels, sampleImgW, sampleImgH
        infoString = json.dumps({ \
            'models': allModels, \
            'res': { 'w': sampleImgW, 'h' : sampleImgH} \
            })
        self.finish(infoString)

def fstWorker(sampleImg, checkpoint_dir, device_t='/gpu:0'):
    modelName = os.path.basename(checkpoint_dir)
    print("Starting worker with model "+modelName+" on GPU "+device_t + "...")

    def printWorker(msg):
         print(str(timestampMs())+" [gpu-worker-"+ modelName+"] " + msg)
    img_shape = get_img(sampleImg).shape

    g = tf.Graph()
    batch_size = 1 # min(len(paths_out), batch_size)
    curr_num = 0
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        while not quit:
            printWorker("Waiting for requests...")
            fileId = requestQueue.get()
            if fileId == "quit-"+modelName:
                printWorker("Received quit command")
                break
            
            t1 = timestampMs()
            printWorker("Received request for style transfer: "+fileId)
            path = os.path.join(uploadPath, fileId)

            printWorker("Reading image "+path)
            img = get_img(path)

            printWorker("Running style transfer...")
            X = np.zeros(batch_shape, dtype=np.float32)
            X[0] = img
            _preds = sess.run(preds, feed_dict={img_placeholder:X})

            pathOut = os.path.join(resultsPath, fileId+".png")
            save_img(pathOut, _preds[0])
            t2 = timestampMs()

            printWorker("Saved result at "+pathOut)
            printWorker("Processing took "+str(t2-t1)+"ms")

    printWorker("Completed")

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--in-path', type=str,
                        dest='in_path',help='dir or file to transform',
                        metavar='IN_PATH', required=True)

    help_out = 'destination (dir or file) of transformed file or files'
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        required=True)

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions', 
                        help='allow different image dimensions')

    return parser

def check_opts(opts):
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'out dir not found!')
        assert opts.batch_size > 0

####
def signal_handler(signum, frame):
    global is_closing
    print('Received stop signal, exiting...')
    tornado.ioloop.IOLoop.instance().stop()
    quit = True

def main():
    global allModels, sampleImg, sampleImgW, sampleImgH
    signal.signal(signal.SIGINT, signal_handler)
    allModels = ["mixed-media-7"] #os.listdir(modelsDir)

    # TODO: this can be expanded to utilize more than one GPU
    nGpus = 1
    workers = {}
    nWorkers = len(allModels)
    sampleImgName = 'sample420x236.jpg'
    # sampleImgName = 'sample840x560.jpg'
    # sampleImgName = 'sample420x280.jpg'
    sampleImg = os.path.join(rootPath, sampleImgName)
    pat = '\D*(?P<w>[0-9]+)x(?P<h>[0-9]+).jpg'
    r  = re.compile(pat)
    m = r.match(sampleImg)
    if m:
        sampleImgW = int(m.group('w'))
        sampleImgH = int(m.group('h'))

    for i in range(0,nWorkers):
        modelName = allModels[i]
        checkpoint = os.path.join(modelsDir, modelName)
        worker = Thread(target=fstWorker, args=(sampleImg, checkpoint, ))
        worker.start()
        workers[modelName] = worker

    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
    
    print("Will terminate GPU workers...")

    for k in workers:
        requestQueue.put("quit-"+str(k))

if __name__ == "__main__":
    main()
