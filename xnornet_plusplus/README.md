# xnornet++

Code for training binary networks on imagenet. 
Xnornet algorithm(https://arxiv.org/abs/1603.05279) is one of the most cited works for training a quantized neural network with 1 bit weight and 1 bit activation. There are numerous work on quantization of neural networks that refer xnornet. Xnornet++(https://arxiv.org/abs/1909.13863) is one such work. As the name suggesets, it is an improvement over the xnornet in terms of accuracy. This code simulates the xnornet++ paper. 
Binarized networks are infamous for not being very easy to get the network converge. Also with the experiments I carried out, I found that design choices for ex. ordering of batch norm, convolution and relu is crucial to extract the best possible accuracy. Unfortunately there are not many published works on binary networks with resnet which makes it even more difficult to get this running. Please cite this repo if you use it for your work. Would really appreciate it and will continue to release more work in future.
Hopefully this code will help you get started with network quantization

## Accuracy on Imagenet Validation for Resnet-18
|   Network   | Top-1(%) |
|-------------|----------|
| Real Valued |     69.3 |
| Paper       |     57.1 |
| This repo   |     55.8 |
| This repo*  |     57.6 |

 *Using real valued downsample, but weights are binarized
## Dependencies

 - Pytorch - 1.5.0 
 - TensorboardX
   pip install tensorboardx 
 

## Training

Download imagenet dataset 
Run the following command to train a resnet-18 network 
    python main.py path_to_imagenet_dataset -a resnet18_preact_bin --lr 1e-1 --weight-decay 1e-5 --workers 16 --model_dir data/binary_out --batch-size 256 --quantize --optimizer sgd 

##Notes
I have used cosine learning decay with sgd that enables this repo to get trained in 64 epochs. Normally binary networks are trained with adam using step learning rate decay which requires approx. 90 epoch to reach final accuracy. 
Also fixed a bug because of which one block was not getting binarized and therefore was getting much highter accuracy before. 
## Acknowledgement

[https://github.com/jiecaoyu/XNOR-Net-PyTorch](https://github.com/jiecaoyu/XNOR-Net-PyTorch)

[https://github.com/pytorch/examples/blob/0.3.1/imagenet/main.py](https://github.com/pytorch/examples/blob/0.3.1/imagenet/main.py)
