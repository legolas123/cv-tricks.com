1. First step is to download the models:

In linux you can download the model by running this script: 
    
    sh get_models.sh

Windows users can download the models here:
 https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt
 http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel
 https://github.com/richzhang/colorization/raw/master/colorization/resources/pts_in_hull.npy



2. Now, you can colorize your images using this command:

     python colorize_image.py --prototxt colorization_deploy_v2.prototxt --caffemodel colorization_release_v2.caffemodel --     kernel pts_in_hull.npy  --input input.png


