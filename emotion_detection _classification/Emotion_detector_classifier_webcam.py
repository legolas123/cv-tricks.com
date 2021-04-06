# importing the module
import cv2
import random
import numpy as np

import mtcnn


from tensorflow.keras.models import model_from_json


cap = cv2.VideoCapture(0) 

detector = mtcnn.MTCNN()



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

print(loaded_model.__class__)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


emotions = ['anger','fear','happy','neutral','sad']

while 1:

    ret, img = cap.read()


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    

    height = img.shape[0]
    width = img.shape[1]


    faces = detector.detect_faces(img)

    if (len(faces)>0):

        for f in faces:
            x,y,w,h = f['box']
            cv2.rectangle(img,(x,y),(x+w,y+h),(,0,0),2)

            roi_gray = gray[y:y+h, x:x+w]

            roi_gray = cv2.resize(roi_gray, (224, 224))

            face = cv2.cvtColor(roi_gray,cv2.COLOR_GRAY2RGB)

            
            # converting image data into proper shape for tensorflow model

            face = np.expand_dims(face, axis=0) 

            emote = loaded_model.predict(face)[0]  ## check face image with model

            emote = np.argmax(emote)


            cv2.putText(img,emotions[emote],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,)

    cv2.imshow('img',img)    


    k = cv2.waitKey(30) & 0xff 
    if k == ord('x'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()








