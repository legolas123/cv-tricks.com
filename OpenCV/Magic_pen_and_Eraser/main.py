import cv2
import numpy as np
import time
import tkinter as tk

load_from_disk = True
if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

pen_img = cv2.resize(cv2.imread('pen.png',1), (50, 50))
eraser_img = cv2.resize(cv2.imread('eraser.png',1), (50, 50))

kernel = np.ones((5,5),np.uint8)

# Making window size adjustable
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# This is the canvas on which we will draw upon
canvas=None

# Create a background subtractor Object
backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

# This threshold determines the amount of disruption in the background.
background_threshold = 600

# A variable which tells you if you're using a pen or an eraser.
switch = 'Pen' 

# With this variable we will monitor the time between previous switch.
last_switch = time.time()

# Initilize x1,y1 points
x1,y1=0,0

# Threshold for noise
noiseth = 800

# Threshold for wiper, the size of the contour must be bigger than for us to
# clear the canvas 
wiper_thresh = 40000

# A variable which tells when to clear canvas, if its True then we clear the canvas
clear = False

# Variable to pause writing/erasing between teaching
pause = False

# Pen color variable which is by default set to blue
color = [255,0,0]

while(1):
    _, frame = cap.read()
    frame = cv2.flip( frame, 1 )
    
    # Initialize the canvas as a black image
    if canvas is None:
        canvas = np.zeros_like(frame)
    
    # Take the top left of the frame and apply the background subtractor there   
    top_left = frame[0: 50, 0: 50]
    fgmask = backgroundobject.apply(top_left)

    # Note the number of pixels that are white, this is the level of disruption.
    switch_thresh = np.sum(fgmask==255)

    # If the disruption is greater than background threshold and there has
    # been some time after the previous switch then you. can change the
    # object type.
    if switch_thresh>background_threshold and (time.time()-last_switch) > 1:
 
        # Save the time of the switch.
        last_switch = time.time()
         
        if switch == 'Pen':
            switch = 'Eraser'
        else:
            switch = 'Pen'

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # If you're reading from memory then load the upper and lower ranges 
    # from there
    if load_from_disk:
        lower_range = penval[0]
        upper_range = penval[1]
            
    # Threshold the HSV image to get the colors specified in penval array
    if not pause:
        mask = cv2.inRange(hsv, lower_range, upper_range)
    
        # Perform the morphological operations to get rid of the noise
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask = cv2.dilate(mask,kernel,iterations = 2)
        
        # Find Contours.
        contours, hierarchy = cv2.findContours(mask,
        cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        # Make sure there is a contour present and also its size is bigger than 
        # the noise threshold.
        if contours and cv2.contourArea(max(contours,
                                    key = cv2.contourArea)) > noiseth:
                    
            c = max(contours, key = cv2.contourArea)    
            x2,y2,w,h = cv2.boundingRect(c)
            
            # Get the area of the contour
            area = cv2.contourArea(c)
            
            # If there were no previous points then save the detected x2,y2 
            # coordinates as x1,y1. 
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2
                
            else:
                if switch == 'Pen':
                    # Draw the line on the canvas
                    canvas = cv2.line(canvas, (x1,y1),
                    (x2,y2), color, 5)
                    
                else:
                    cv2.circle(canvas, (x2, y2), 20,
                    (0,0,0), -1)
            
            # After the line is drawn the new points become the previous points.
            x1,y1= x2,y2
            
            # Now if the area is greater than the wiper threshold then set the  
            # clear variable to True and warn User.
            if area > wiper_thresh:
                cv2.putText(canvas,'Clearing Canvas', (100,200), 
                cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,255), 5, cv2.LINE_AA)
                clear = True 

        else:
            # If there were no contours detected then make x1,y1 = 0
            x1,y1 =0,0
        
        # Switch the images depending upon what we're using, pen or eraser.
        if switch != 'Pen':
            cv2.circle(frame, (x1, y1), 20, (255,255,255), -1)
            frame[0: 50, 0: 50] = eraser_img
        else:
            frame[0: 50, 0: 50] = pen_img

    # Now this piece of code is just for smooth drawing. (Optional)
    _ , mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 
    255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
    background = cv2.bitwise_and(frame, frame,
    mask = cv2.bitwise_not(mask))
    frame = cv2.add(foreground,background)

    # Show image
    cv2.imshow('image',frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
    # if the user presses s then he is prompted for the name of canvas to be saved
    elif k == ord('s'):
        root= tk.Tk()

        canvas1 = tk.Canvas(root, width = 800, height = 500)
        canvas1.pack()

        entry1 = tk.Entry (root) 
        canvas1.create_window(200, 140, window=entry1)

        def getFileName():  
            x1 = entry1.get()
            np.save(f"{x1}.npy", canvas)
            label1 = tk.Label(root, text="file saved succesfully")
            canvas1.create_window(200, 230, window=label1)
            
        button1 = tk.Button(text='Submit name for canvas', command=getFileName)
        canvas1.create_window(200, 180, window=button1)

        root.mainloop()
    
    # If the user presses l then he is prompted for the name of canvas to be loaded
    elif k == ord('l'):
        root= tk.Tk()

        canvas1 = tk.Canvas(root, width = 800, height = 500)
        canvas1.pack()

        entry1 = tk.Entry (root) 
        canvas1.create_window(200, 140, window=entry1)

        def getFileNameLoad():
            global canvas  
            x1 = entry1.get()
            canvas = np.load(f"{x1}.npy")
            label1 = tk.Label(root, text="File opened, you may close this or re enter name if you want to open new file.")
            canvas1.create_window(200, 230, window=label1)
            root.destroy()
            
        button1 = tk.Button(text='Submit name to load canvas', command=getFileNameLoad)
        canvas1.create_window(200, 180, window=button1)

        root.mainloop()
    
    # If the user presses p then he is allowed to pause his writing/erasing and explain what
    # he has written
    elif k == ord('p'):
        pause = not pause

        # this is required or else a strange line is encountered between last point where pen was before pausing
        # and the new point where pen is after continuing to write.
        x1,y1=0,0

    # If user presses c then he is allowed to change pen color which is by default set to blue.
    elif k == ord('c'):
        root= tk.Tk()

        canvas1 = tk.Canvas(root, width = 500, height = 300)
        canvas1.pack()

        entry1 = tk.Entry (root)
          
        canvas1.create_window(200, 140, window=entry1)
        canvas1.create_text(250,20,fill="darkblue",font="Times 10 italic bold",
                        text="Enter RGB values seperated with commas but no space to change pen color")
        def changeColor():
            global color  
            global canvas
            x1 = entry1.get()

            try:
                x1 = x1.split(",")
                r = int(x1[0])
                g = int(x1[1])
                b = int(x1[2])

            except:
                label1 = tk.Label(root, text="Color values must be similar to the following example\n200,200,0")
                canvas1.create_window(200, 230, window=label1)
                return
            
            # all color values must be between 0-255
            if (r > 255 or r < 0) or (g > 255 or g < 0) or (b > 255 or b < 0):
                label1 = tk.Label(root, text="Color values must be between 0-255")
                canvas1.create_window(200, 230, window=label1)

            else:
                color = [r,g,b]
                label1 = tk.Label(root, text="Color changed succesfully")
                canvas1.create_window(200, 230, window=label1)
                root.destroy()
            
        button1 = tk.Button(text='Enter', command=changeColor)
        canvas1.create_window(200, 180, window=button1)

        root.mainloop()

    # Clear the canvas after 1 second if the clear variable is true
    if clear == True:
        
        time.sleep(1)
        canvas = None
        
        # And then set clear to false
        clear = False
        
cv2.destroyAllWindows()
cap.release()