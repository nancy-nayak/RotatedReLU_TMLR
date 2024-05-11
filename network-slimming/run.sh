#####   NO REGULARIZER =================
python main.py --dataset cifar10 --arch resnet --save './logs_resnet164' --depth 164
# python main.py --dataset cifar10 -e --prune "./logs_resnet164/model_best.pth.tar" --arch resnet --depth 164


#####   WITH REGULARIZER =================
python main.py -sr --s 0.00001 --dataset cifar10 --arch resnet --save './logs_resnet164reg' --depth 164
# python main.py --dataset cifar10 -e --prune "./logs_resnet164reg/model_best.pth.tar" --arch resnet --depth 164
