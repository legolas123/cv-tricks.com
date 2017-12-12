from network import VGG16
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-image", "--image", type=str, default='test.jpg',help="Path of test image")
ap.add_argument("-num_class","--class",type=int, default=2,help="(required) number of classes to be trained")
args = vars(ap.parse_args())


base_model = VGG16.VGG16(include_top=False, weights=None)
x = base_model.output
x = Dense(128)(x)
x = GlobalAveragePooling2D()(x)
predictions = Dense(args["class"], activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights("cv-tricks_fine_tuned_model.h5")

inputShape = (224,224) # Assumes 3 channel image
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)   # shape is (224,224,3)
image = np.expand_dims(image, axis=0)  # Now shape is (1,224,224,3)

image = image/255.0

preds = model.predict(image)
print(preds)

