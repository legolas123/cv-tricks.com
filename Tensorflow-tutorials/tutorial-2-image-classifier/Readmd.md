This code assumes that the training data is in a folder called training data in this format:
training_data
     -dogs
     -cats

Similarly, we need the validation dataset in a similar format:

testing_data
     -dogs
     -cats

Minimum number of images for having an effective training will be 2000 images per class, which can be downloaded from here: 
https://drive.google.com/open?id=0B2L-gJqoC67TTG5CX1ozbG5jTE0

You can also download the complete dataset of Kaggle dog-cat challenge. 

1. In order to start the training: 

python train.py --train training_data/ --val testing_data/ --num_classes 2

2. Once your model is trained, it will save the model files in the current folder. 

In order to predict using the trained model, we can run the predict.py code by passing it our test image. 

python predict.py testing_data/cats/cat.1110.jpg


