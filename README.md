Rotate the ReLU to Sparsify Deep Networks Implicitly
===============================================================================

*Nancy Nayak, Sheetal Kalyani*.\
TMLR 2024

## Abstract 
Compact and energy-efficient models have become essential in this era when deep learning-based solutions are widely used for various real-life tasks. 
In this paper, we propose rotating the ReLU activation to give an additional degree of freedom in conjunction with the appropriate initialization of 
the rotation. This combination leads to implicit sparsification without the use of a regularizer. We show that this rotated ReLU (RReLU) activation improves 
the representation capability of the parameters/filters in the network and eliminates those parameters/filters that are not crucial for the task, giving rise 
to significant savings in memory and computation. While the state-of-the-art regularization-based Network-Slimming method achieves $32.33\%$ saving in memory 
and $26.38\%$ saving in computation with ResNet-164, RReLU achieves a saving of $35.92\%$ in memory and $25.97\%$ in the computation with a better accuracy. 
The savings in memory and computation further increase by $64.67\%$ and $52.96\%$, respectively, with the introduction of $L_1$ regularization to the RReLU slopes.
We note that the slopes of the rotated ReLU activations act as coarse feature extractors and can eliminate unnecessary features before retraining. Our studies indicate 
that features always choose to pass through a lesser number of filters. We demonstrate the results with popular datasets such as MNIST, CIFAR-10, CIFAR-100, SVHN,
and Imagenet with different architectures, including Vision Transformers and EfficientNet. We also briefly study the impact of adversarial attacks on RReLU-based ResNets
and observe that we get better adversarial accuracy for the architectures with RReLU than ReLU. We also demonstrate how this concept of rotation can be applied to the GELU
and SiLU activation functions, commonly utilized in Transformer and EfficientNet architectures, respectively. The proposed method can be utilized by combining with other 
structural pruning methods resulting in better sparsity. For the GELU-based multi-layer perceptron (MLP) part of the Transformer, we obtain 2.6\% improvement in accuracy
with 6.32\% saving in both memory and computation.

## Installation 
Install the required packages by running the following command (stay inside the directory codes_ICCV2023)
``` bash
conda env create -f rrelujan.yml
```
and activate them
``` bash
conda activate rrelujan
```

## Instructions to obtain the results in Table 1. For ResNets change your directory to pytorch_resnet_cifar10.
## Copy-paste the corresponding commands to the run.sh file inside the directory and run.

1. To train the ResNet20 architecture with ReLU and CIFAR10 dataset:
``` bash
for model in resnet20
do
	python -u trainer_adv_Jan.py \
          --arch=$model\
          --dataset=CIFAR10 \
          --epoch=1200 \
          --save-dir=resultsICCV/save_standard_$model
done
```
2. To train the ResNet20 architecture with ReLU and CIFAR100 dataset:
``` bash
for model in resnet20_f
do
	python -u trainer_adv_Jan.py \
          --arch=$model\
          --dataset=CIFAR100 \
          --epoch=1200 \
          --save-dir=resultsICCV/save_standard_$model
done
```

3. To train the ResNet20 architecture with RReLU and CIFAR10 dataset:
``` bash
for model in resnet20_rotatedrelu_maam
do
	python -u trainer_adv_Jan.py \
          --arch=$model\
          --dataset=CIFAR10 \
          --epoch=1200 \
          --save-dir=resultsICCV/save_rrelu_$model
done
```
4. To train the ResNet20 architecture with RReLU and CIFAR100 dataset:
``` bash
for model in resnet20_rotatedrelu_maam_f
do
	python -u trainer_adv_Jan.py \
          --arch=$model\
          --dataset=CIFAR100 \
          --epoch=1200 \
          --save-dir=resultsICCV/save_rrelu_$model
done
```
5. To test the model and to find out how many filters are inactive and layerwise inactive filters for the ResNet20 architecture with RReLU and CIFAR10 dataset:
Before testing, make sure that you have the fully trained model saved inside the directory ./pytorch_resnet_cifar10/resultsICCV/save_rrelu_resnet20_rotatedrelu_maam
By changing the value of zeta, you can have different level of pruning. 
Find a zeta that has no/negligible degradation in performance using validation set i.e. validation=1.
``` bash
for model in resnet20_rotatedrelu_maam
do
	python -u trainer_adv_Jan.py \
          --arch=$model -e \
          --prune="./resultsICCV/save_rrelu_resnet20_rotatedrelu_maam/checkpoint_best.th" \
          --dataset=CIFAR10 \
          --zeta=0.1 \
          --validation=1 \
          --epoch=1200 \
          --save-dir=resultsICCV/save_rrelu_eval_$model
done
```
Then test using that zeta after using validation=0
``` bash
for model in resnet20_rotatedrelu_maam
do
	python -u trainer_adv_Jan.py \
          --arch=$model -e \
          --prune="./resultsICCV/save_rrelu_resnet20_rotatedrelu_maam/checkpoint_best.th" \
          --dataset=CIFAR10 \
          --zeta=0.1 \
          --validation=0 \
          --epoch=1200 \
          --save-dir=resultsICCV/save_rrelu_eval_$model
done
```
6. From the list of filters alive along the depth, you can calculate the number of parameters and FLOPs by running the following. Enter the list of alive
filters obtained from the above command to _listrrelu in count.py and count_preactivation.py

For ResNet20, ResNet56, WRN-40-4, WRN-16-4, ResNet-50 run
``` bash
python count.py
```
For ResNet-110-pre and ResNet-164-pre, run
``` bash
python count_preactivation.py
```

7. Figure 5.a and Figure 5.b which are our baselines are reproduced from the code provided by the authors. 
Go to directory network-slimming and run appropriate code from run.sh

## Citation 

If you find our work useful in your research, please consider citing:

```bash
@article{
    anonymous2023rotate,
    title    = {Rotate the Re{LU} to Sparsify Deep Networks Implicitly},
    author   = {Anonymous},
    journal  = {Submitted to Transactions on Machine Learning Research},
    year     = {2023},
    url      = {https://openreview.net/forum?id=Nzy0XmCPuZ},
    note     = {Under review}
}
```
