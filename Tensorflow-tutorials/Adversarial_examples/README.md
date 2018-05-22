# This code can be used to generate the adversarial image for any given image. Then, we also run that script to show the output of adversarial image on InceptionV3 architecture
# Run this command to run on a new image called 1.jpg
python classify_image.py --image_file 1.jpg 

# If image file is not specified, it runs on imagenet/cropped_panda.jpg. 
# Adversarial image is stored in the folder adver_images.

# By default it runs step_targeted_attack method for adversarial image geenration. There are two more methods step_ll_adversarial_images and step_fgsm which can be used for generating adversarial images. 

# To run other two methods, comment the line with call to step_targeted_attack(line 132) and Uncomment the appropriate methods below like step_fgsm(line 135).  
