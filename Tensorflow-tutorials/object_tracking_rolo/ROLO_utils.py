
import os, sys, time
import numpy as np
import cv2
import tensorflow as tf
import math
import pickle

class ROLO_utils:
        def __init__(self):
            print("Intialised utilities")

        # Not Face user
        def file_to_img(self, filepath):
            print('Processing '+ filepath)
            img = cv2.imread(filepath)
            return img


        def file_to_video(self, filepath):
            print('processing '+ filepath)
            try:
                    video = cv2.VideoCapture(filepath)
            except IOError:
                    print('cannot open video file: ' + filepath)
            else:
                    print('unknown error reading video file')
            return video


        def iou(self,box1,box2):
                tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
                lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
                if tb < 0 or lr < 0 : intersection = 0
                else : intersection =  tb*lr
                return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

        def load_folder(self, path):
                paths = [os.path.join(path,fn) for fn in next(os.walk(path))[2]]
                return paths


        def find_best_location(self, locations, gt_location):
                # locations (class, x, y, w, h, prob); (x, y) is the middle pt of the rect
                # gt_location (x1, y1, w, h)
                x1 = gt_location[0]
                y1 = gt_location[1]
                w = gt_location[2]
                h = gt_location[3]
                gt_location_revised= [x1 + w/2, y1 + h/2, w, h]

                max_ious= 0
                for location, id in enumerate(locations):
                        location_revised = location[1:5]
                        ious = self.iou(location_revised, gt_location_revised)
                        if ious >= max_ious:
                                max_ious = ious
                                index = id
                return locations[index]


        def save_yolo_output(self, out_fold, yolo_output, filename):
                name_no_ext= os.path.splitext(filename)[0]
                output_name= name_no_ext + ".yolo"
                path = os.path.join(out_fold, output_name)
                pickle.dump(yolo_output, open(path, "rb"))


        def load_yolo_output(self, fold, batch_size, num_steps, step):
                paths = [os.path.join(fold,fn) for fn in next(os.walk(fold))[2]]
                paths = sorted(paths)
                st= step*batch_size*num_steps
                ed= (step+1)*batch_size*num_steps
                paths_batch = paths[st:ed]

                yolo_output_batch= []
                ct= 0
                for path in paths_batch:
                        ct += 1
                        #yolo_output= pickle.load(open(path, "rb"))
                        yolo_output = np.load(path)
                    
                        yolo_output= np.reshape(yolo_output, 4102)
                        yolo_output[4096]= 0
                        yolo_output[4101]= 0
                        yolo_output_batch.append(yolo_output)
                print(yolo_output_batch)
                yolo_output_batch= np.reshape(yolo_output_batch, [batch_size*num_steps, 4102])
                return yolo_output_batch


        def load_yolo_output_test(self, fold, batch_size, num_steps, id):
                paths = [os.path.join(fold,fn) for fn in next(os.walk(fold))[2]]
                paths = sorted(paths)
                st= id
                ed= id + batch_size*num_steps
                paths_batch = paths[st:ed]

                yolo_output_batch= []
                ct= 0
                for path in paths_batch:
                        ct += 1
                        yolo_output = np.load(path)
                        #print(yolo_output)
                        yolo_output= np.reshape(yolo_output, 4102)
                        yolo_output_batch.append(yolo_output)
                yolo_output_batch= np.reshape(yolo_output_batch, [batch_size*num_steps, 4102])
                return yolo_output_batch


def createFolder( path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_folder( path):
        paths = [os.path.join(path,fn) for fn in next(os.walk(path))[2]]
        return sorted(paths)


def find_yolo_location( fold, id):
        paths = [os.path.join(fold,fn) for fn in next(os.walk(fold))[2]]
        paths = sorted(paths)
        path= paths[id-1]
        #print(path)
        yolo_output = np.load(path)
        #print(yolo_output[0][4096:4102])
        yolo_location= yolo_output[0][4097:4101]
        return yolo_location


def find_rolo_location( fold, id):
        filename= str(id) + '.npy'
        path= os.path.join(fold, filename)
        rolo_output = np.load(path)
        return rolo_output


def file_to_img( filepath):
    img = cv2.imread(filepath)
    return img

def locations_normal(wid, ht, locations):
    #print("location in func: ", locations)
    wid *= 1.0
    ht *= 1.0
    locations[0] *= wid
    locations[1] *= ht
    locations[2] *= wid
    locations[3] *= ht
    return locations


def debug_location( img, location):
    img_cp = img.copy()
    x = int(location[1])
    y = int(location[2])
    w = int(location[3])//2
    h = int(location[4])//2
    cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
    cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
    cv2.putText(img_cp, str(location[0]) + ' : %.2f' % location[5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    cv2.imshow('YOLO_small detection',img_cp)
    cv2.waitKey(1)

def debug_2_locations( img, rolo_location, yolo_location):
    img_cp = img.copy()
    for i in range(2):  # b-g-r channels
        if i== 0: location= rolo_location; color= (255, 0, 0)       # blue for rolo
        elif i ==1: location= yolo_location; color= (0,255,0)   # green for yolo
        x = int(location[0])
        y = int(location[1])
        w = int(location[2])
        h = int(location[3])
        if i == 1: cv2.rectangle(img_cp,(x-w//2, y-h//2),(x+w//2,y+h//2), color, 2)
        elif i== 0: cv2.rectangle(img_cp,(x-w//2, y-h//2),(x+w//2,y+h//2), color, 2)
    cv2.imshow('2 locations',img_cp)
    cv2.waitKey(100)
    return img_cp

def save_rolo_output(out_fold, rolo_output, filename):
    name_no_ext= os.path.splitext(filename)[0]
    output_name= name_no_ext
    path = os.path.join(out_fold, output_name)
    np.save(path, rolo_output)


def save_rolo_output_test( out_fold, rolo_output, step, num_steps, batch_size):
        assert(len(rolo_output)== batch_size)
        st= step - 2 #* batch_size * num_steps
        for i in range(batch_size):
            id = st + (i + 1)* num_steps + 1
            pred = rolo_output[i]
            path = os.path.join(out_fold, str(id))
            np.save(path, pred)



def locations_from_0_to_1(wid, ht, locations):
    #print("location in func: ", locations[0][0])
    wid *= 1.0
    ht *= 1.0
    for i in range(len(locations)):
        # convert top-left point (x,y) to mid point (x, y)
        locations[i][0] += locations[i][2] / 2.0
        locations[i][1] += locations[i][3] / 2.0
        # convert to [0, 1]
        locations[i][0] /= wid
        locations[i][1] /= ht
        locations[i][2] /= wid
        locations[i][3] /= ht
    return locations


def validate_box(box):
    for i in range(len(box)):
        if math.isnan(box[i]): box[i] = 0


def iou(box1, box2):
    # Prevent NaN in benchmark results
    validate_box(box1)
    validate_box(box2)

    # change float to int, in order to prevent overflow
    box1 = list(map(int, box1))
    box2 = list(map(int, box2))

    tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
    lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
    if tb <= 0 or lr <= 0 :
        intersection = 0
    else : intersection =  tb*lr
    return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

def iou_0_1(box1, box2, w, h):
    box1 = locations_normal(w,h,box1)
    box2 = locations_normal(w,h,box2)
    #print box1
    #print box2
    return iou(box1,box2)

def cal_rolo_IOU(location, gt_location):
    location[0] = location[0] - location[2]/2
    location[1] = location[1] - location[3]/2
    loss = iou(location, gt_location)
    return loss


def cal_yolo_IOU(location, gt_location):
    # Translate yolo's box mid-point (x0, y0) to top-left point (x1, y1), in order to compare with gt
    location[0] = location[0] - location[2]/2
    location[1] = location[1] - location[3]/2
    loss = iou(location, gt_location)
    return loss

def load_yolo_output_test(fold, batch_size, num_steps, id):
        paths = [os.path.join(fold,fn) for fn in next(os.walk(fold))[2]]
        paths = sorted(paths)
        st= id
        ed= id + batch_size*num_steps
        paths_batch = paths[st:ed]

        yolo_output_batch= []
        ct= 0
        for path in paths_batch:
                ct += 1
                yolo_output = np.load(path)
                #print(yolo_output)
                yolo_output= np.reshape(yolo_output, 4102)
                yolo_output_batch.append(yolo_output)
        yolo_output_batch= np.reshape(yolo_output_batch, [batch_size*num_steps, 4102])
        return yolo_output_batch
