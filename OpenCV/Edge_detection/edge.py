import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(
        description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                    'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                    'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--write_video', help='Do you want to write the output video', default=False)
parser.add_argument('--prototxt', help='Path to deploy.prototxt',default='deploy.prototxt', required=False)
parser.add_argument('--caffemodel', help='Path to hed_pretrained_bsds.caffemodel',default='hed_pretrained_bsds.caffemodel', required=False)
parser.add_argument('--width', help='Resize input image to a specific width', default=500, type=int)
parser.add_argument('--height', help='Resize input image to a specific height', default=500, type=int)
parser.add_argument('--savefile', help='Specifies the output video path', default='output.mp4', type=str)
args = parser.parse_args()

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

cv.dnn_registerLayer('Crop', CropLayer)

# Load the model.
net = cv.dnn.readNet(args.prototxt, args.caffemodel)

## Create a display window
kWinName = 'Holistically-Nested_Edge_Detection'
cv.namedWindow(kWinName, cv.WINDOW_AUTOSIZE)

cap = cv.VideoCapture(args.input if args.input else 0)

if args.write_video:
    # Define the codec and create VideoWriter object
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(w,h)
    # w, h = args.width,args.height
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    writer = cv.VideoWriter(args.savefile, fourcc, 25, (w, h))
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    inp = cv.dnn.blobFromImage(frame, scalefactor=1.0, size=(args.width, args.height),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv.resize(out, (frame.shape[1], frame.shape[0]))
    out = 255 * out
    out = out.astype(np.uint8)
    out=cv.cvtColor(out,cv.COLOR_GRAY2BGR)
    con=np.concatenate((frame,out),axis=1)
    if args.write_video:
        writer.write(np.uint8(con))
    cv.imshow(kWinName,con)
    