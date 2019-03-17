https://www.youtube.com/watch?v=b6pomaq30Gg
To Run MPI:
python3 run.py --input runner.jpeg --proto pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt  --model pose/mpi/pose_iter_160000.caffemodel --dataset MPI

To Run Body_25 model:
python3 run.py --input runner.jpeg --proto pose/body_25/body_25_deploy.prototxt  --model pose/body_25/pose_iter_584000.caffemodel


To Run COCO:
python3 run.py --input runner.jpeg --proto pose/coco/deploy_coco.prototxt  --model pose/coco/pose_iter_440000.caffemodel --dataset COCO

on video being tested:
python3 run_on_video_webcam2.py --output out.mp4 --input input.mp4 

