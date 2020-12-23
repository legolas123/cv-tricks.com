
import sys
import ROLO_utils as utils
from YOLO_network import YOLO
from ROLO_network import ROLO
import tensorflow as tf
import cv2
import argparse
import numpy as np
import os
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument("-p" , "--path", required = True, type = str, help = "path to video or image folder")
parser.add_argument("-v", '--video',action='store_true')
args = vars(parser.parse_args())
start_time=time.time()
video = args['video']
path1 = args['path']
if video:
    path = os.path.dirname(path1)
    img_fold_path = os.path.join(path, 'img')
else:
    path = os.path.split(path1)[0]
    img_fold_path = path1

yolo = YOLO(path1, video=video)
yolo.run_net()


tf.reset_default_graph()
rolo = ROLO(path1, video = video)
width, height, num_steps = rolo.run_net()

yolo_out_path= os.path.join(path, 'yolo_out')
rolo_out_path= os.path.join(path, 'rolo_out_test')

paths_imgs = utils.load_folder(img_fold_path)
paths_rolo= utils.load_folder(rolo_out_path)

utils.createFolder(os.path.join(path, 'output/frames'))
utils.createFolder(os.path.join(path, 'output/videos'))

fourcc= cv2.VideoWriter_fourcc(*'DIVX')
video_name = 'test_video.avi'
video_path = os.path.join(os.path.join(path, 'output/videos'), video_name)
video = cv2.VideoWriter(video_path, fourcc, 20, (width, height))

for i in range(len(paths_rolo)- num_steps):
        id= i + 1
        test_id= id + num_steps - 2  #* num_steps + 1

        path2 = paths_imgs[test_id]
        img = utils.file_to_img(path2)

        if(img is None): break

        yolo_location= utils.find_yolo_location(yolo_out_path, test_id)
        yolo_location= utils.locations_normal( width, height, yolo_location)
        print(yolo_location)

        rolo_location= utils.find_rolo_location( rolo_out_path, test_id)
        rolo_location = utils.locations_normal( width, height, rolo_location)
        print(rolo_location)

        frame = utils.debug_2_locations( img, rolo_location, yolo_location)
        video.write(frame)

        frame_name= os.path.join(os.path.join(path, 'output/frames'),str(test_id)+'.jpg')
        cv2.imwrite(frame_name, frame)

video.release()
cv2.destroyAllWindows()

