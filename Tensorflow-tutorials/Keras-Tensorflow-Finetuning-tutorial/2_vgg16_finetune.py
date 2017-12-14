from network import VGG16
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras import optimizers 
from keras.callbacks import TensorBoard,ModelCheckpoint
import argparse
from time import time


#################################################################################################################################
#################################################################################################################################
# Following steps are required to fine-tune the model
#
#  1. Specify the path to training and testing data, along with number of classes and image size.
#  2. Do some random image transformations to increase the number of training samples and load the training and testing data
#  3. Create VGG16 network graph(without top) and don't load any weights
#  4. Add the top based on number of classes we have. 
#  5. Load the weights saved in pre-training phase. Specify the optimizer, loss etc and start the training
##################################################################################################################################
##################################################################################################################################

##### Step-1:
############ Specify path to training and testing data. Minimum 100 images per class recommended.
############ Default image size is 160    
img_size=224
ap = argparse.ArgumentParser()
ap.add_argument("-train","--train_dir",type=str, required=True,help="(required) the train data directory")
ap.add_argument("-num_class","--class",type=int, default=2,help="(required) number of classes to be trained")
ap.add_argument("-val","--val_dir",type=str, required=True,help="(required) the validation data directory")

args = vars(ap.parse_args())

##### Step-2:
############ Do some random image transformations to increase the number of training samples
############ Note that we are scaling the image to make all the values between 0 and 1. That's how our pretrained weights have been done too
############ Default batch size is 8 but you can reduce/increase it depending on how powerful your machine is. 

batch_size=10

train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
        args["train_dir"],
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        args["val_dir"],
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='categorical')


##### Step-3:
############ Create VGG-16 network graph without the last layers and load imagenet pretrained weights
############ Default image size is 160    
print('loading the model and the pre-trained weights...')

base_model = VGG16.VGG16(include_top=False, weights=None)
## Here we will print the layers in the network
i=0
for layer in base_model.layers:
    layer.trainable = True
    i = i+1
    print(i,layer.name)
#sys.exit()

##### Step-4:
############ Add the top as per number of classes in our dataset
############ Note that we are using Dropout layer with value of 0.2, i.e. we are discarding 20% weights
############

x = base_model.output
x = Dense(128)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(args["class"], activation='softmax')(x)


##### Step-5:
############ Specify the complete model input and output, optimizer and loss

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

filepath = 'cv-tricks_fine_tuned_model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,save_best_only=False,save_weights_only=False, mode='auto',period=1)
callbacks_list = [checkpoint,tensorboard]


model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights("cv-tricks_pretrained_model.h5")

model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),metrics=["accuracy"])

num_training_img=1000
num_validation_img=400
stepsPerEpoch = num_training_img/batch_size
validationSteps= num_validation_img/batch_size
model.fit_generator(
        train_generator,
        steps_per_epoch=stepsPerEpoch,
        epochs=20,
        callbacks = callbacks_list,
        validation_data = validation_generator,
        validation_steps=validationSteps
        )


