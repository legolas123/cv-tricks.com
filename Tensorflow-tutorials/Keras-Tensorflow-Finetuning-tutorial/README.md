#1. Start pretraining by running this command:
python 1_vgg16_pretrain.py  -train ../tutorial-2-image-classifier/training_data/ -val ../tutorial-2-image-classifier/testing_data/ -num_class 2

#Now, a weights file with the name cv-tricks_pretrained_model.h5 should be saved. Now, in order to start fine-tuning process, run this command
python 2_vgg16_finetune.py  -train ../tutorial-2-image-classifier/training_data/ -val ../tutorial-2-image-classifier/testing_data/ -num_class 2

#This will save the finetuned weights(cv-tricks_fine_tuned_model.h5) with 98%+ accuracy

#Now, you can use the predict script to run this trained model on a new image of a dog or cat
python 3_predict.py --image test.jpg

## This prints a list which contains the probabilities for image being cat or dog.
#[ prob of cat  , prob of dog]
