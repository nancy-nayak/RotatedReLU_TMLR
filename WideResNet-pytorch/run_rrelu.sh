# python train.py --dataset cifar10 \
#                 --layers 40 \
#                 --widen-factor 4 \
#                 --name WideResNet-40-4 \
#                 --typer RReLU \
#                 --retrain 0 \
#                 --epochs 400 \
#                 --save-dir runs


# python train.py --dataset cifar10 \
#                 -e \
#                 --prune="./runs/WideResNet-40-4/RReLU/model_best.pth.tar" \
#                 --layers 40 \
#                 --widen-factor 4 \
#                 --name WideResNet-40-4 \
#                 --typer RReLU \
#                 --retrain 0 \
#                 --gamma=0.0 \
#                 --epochs 400 \
#                 --save-dir runs_eval_rrelu



# python train.py --dataset cifar100 \
#                 -e \
#                 --prune="./runs_f/WideResNet-40-4/RReLU/model_best.pth.tar" \
#                 --layers 40 \
#                 --widen-factor 4 \
#                 --name WideResNet-40-4 \
#                 --typer RReLU \
#                 --retrain 0 \
#                 --gamma=0.04 \
#                 --epochs 400 \
#                 --save-dir runs_f_eval_rrelu

# python train.py --dataset cifar100 \
#                 --layers 40 \
#                 --widen-factor 4 \
#                 --name WideResNet-40-4 \
#                 --typer RReLU \
#                 --retrain 0 \
#                 --epochs 400 \
#                 --save-dir runs_f



# ## SVHN, RReLU training,
# python train.py --dataset SVHN \
#                 --layers 16 \
#                 --widen-factor 4 \
#                 --name WideResNet-16-4 \
#                 --typer RReLU \
#                 --lr=0.01 \
#                 --retrain 0 \
#                 --warm_up \
#                 --droprate 0.0 \
#                 --epochs 1000 \
#                 --save-dir runs_svhn  


# ## SVHN, RReLU training, 
# python train.py --dataset SVHN -e \
#                 --layers 16 \
#                 --prune="./runs_svhn/WideResNet-16-4/RReLU/model_best.pth.tar" \
#                 --widen-factor 4 \
#                 --name WideResNet-16-4 \
#                 --typer RReLU \
#                 --retrain 0 \
#                 --warm_up \
#                 --gamma=0.04 \
#                 --droprate 0.0 \
#                 --epochs 1000 \
#                 --save-dir runs_svhn_eval 


# python train.py --dataset cifar100 \
#                 --layers 40 \
#                 --widen-factor 4 \
#                 --name WideResNet-40-4 \
#                 --typer RReLU \
#                 --lr 0.1 \
#                 --retrain 1 \
#                 --epochs 400 \
#                 --save-dir runs_retrained


# python train_test.py --dataset cifar100 \
#                 -e \
#                 --prune="./runs_retrained/WideResNet-40-4/RReLU/model_best.pth.tar" \
#                 --layers 40 \
#                 --widen-factor 4 \
#                 --name WideResNet-40-4 \
#                 --typer RReLU \
#                 --lr 0.001 \
#                 --retrain 0 \
#                 --epochs 300 \
#                 --save-dir runs_retrained_evals

# python train.py --dataset cifar100 -e \
#                 --layers 40 \
#                 --prune="./runs/WideResNet-40-4/RReLU/model_best.pth.tar" \
#                 --widen-factor 4 \
#                 --name WideResNet-40-4 \
#                 --typer RReLU \
#                 --epochs 400 \
#                 --save-dir runs_evals