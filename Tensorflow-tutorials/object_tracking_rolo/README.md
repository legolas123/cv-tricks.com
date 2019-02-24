# ROLO

This repo contains the code for object tracking using ROLO network.
ROLO uses YOLO network for object detection and LSTM for sequential processing. So we need pretrained weights for YOLO and LSTM.

Clone the repo and run the files as mentioned below.

Downloads the pretrained weights and extract them using downloads.sh script 

> `sh download.sh`

ROLO can be run in two different modes. We can give as input either the video file directly or give the folder containing all the frames in the video
To test the ROLO network you can use the sample videos in sample_videos folder or give your own video files.

 - To run ROLO with a video file, execute the following command from repo directory

    

>  python ROLO_test.py --path PATH_TO_VIDEO --v

where PATH_TO_VIDEO is the path to video file

For eg., to run with one of the sample videos,

> python ROLO_test.py --path sample_videos/test_video1.mp4 --video

 - To run ROLO with frames of a video, execute the following command from repo directory

> python ROLO_test.py --path PATH_TO_FRAMES

where PATH_TO_FRAMES is the path to the folder containing the frames of the video

The script first run YOLO network and then ROLO network. We can see the prediction visuals during the run time itself.  When the script completes a tracking video is created in the **output** folder which is located in the same directory as input video or input frames folder.

