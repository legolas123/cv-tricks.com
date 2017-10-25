import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from squeezenet import SqueezeNet
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg


model = SqueezeNet()

img = image.load_img('pexels-photo-280207.jpeg', target_size=(227, 227))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
all_results = decode_predictions(preds)
for results in all_results:
    for result in results:
        print('Probability %0.2f%% => [%s]' % (100*result[2], result[1]))
        #result_text= 'Probability %0.2f%% => [%s]' % (100*result[2], result[1])
        #break
#plt.figure(num=1,figsize=(8, 6), dpi=80)
#plt.imshow(img)
#plt.text(130,90,result_text,horizontalalignment='center', verticalalignment='center',fontsize=16,color='black')
#plt.axis('off')
#plt.show()

