First step is to download the models:

sh get_models.sh

Now, you can colorize your images using this command:

python colorize_image.py --prototxt colorization_deploy_v2.prototxt --caffemodel colorization_release_v2.caffemodel --kernel pts_in_hull.npy  --input input.png


