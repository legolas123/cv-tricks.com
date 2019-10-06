Code for running OpenPose code for human pose estimation using deep learning in OpenCV. 

First step is download the pretrained models:

# On Ubuntu/Mac, run this command to download the models.
sh get_model.sh

Now, we can just run these models using the commands below. There are three pretrained models available each trained on a different dataset. In the blog, we have only shown MPI but you can run the other two as well. 

To Run MPI pretrained model on an image sample.jpg:
python3 run_pose.py --input sample.jpg --proto pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt  --model pose/mpi/pose_iter_160000.caffemodel --dataset MPI

To Run Body_25 pretrained model on an image sample.jpg:
python3 run_pose.py --input sample.jpg --proto pose/body_25/body_25_deploy.prototxt  --model pose/body_25/pose_iter_584000.caffemodel


To Run COCO pretrained model on an image sample.jpg:
python3 run_pose.py --input sample.jpg --proto pose/coco/deploy_coco.prototxt  --model pose/coco/pose_iter_440000.caffemodel --dataset COCO

The output results are shown in a new image result_sample.jpg saved on your hard disk.  
