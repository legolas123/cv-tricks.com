import numpy as np
from keras import applications
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

# build the VGG16 network
model = applications.VGG16(weights='imagenet')
img = image.load_img('pexels-photo-280207.jpeg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
all_results = decode_predictions(preds)
for results in all_results: 
    for result in results:
        print('Probability %0.2f%% => [%s]' % (100*result[2], result[1]))
#print('Predicted:', decode_predictions(preds))
