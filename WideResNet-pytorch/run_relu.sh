python train.py --dataset cifar10 \
                --layers 40 \
                --widen-factor 4 \
                --name WideResNet-40-4 \
                --typer ReLU \
                --retrain 0 \
                --epochs 400 \
                --save-dir runs



python train.py --dataset cifar10 \
                --layers 40 \
                -e \
                --prune="./runs/WideResNet-40-4/ReLU/model_best.pth.tar" \
                --widen-factor 4 \
                --name WideResNet-40-4 \
                --typer ReLU \
                --retrain 0 \
                --gamma=0.0 \
                --epochs 400 \
                --save-dir runs_eval


# python train.py --dataset cifar10 \
#                 --layers 40 \
#                 -e \
#                 --prune="./runs/WideResNet-40-4/ReLU/model_best.pth.tar" \
#                 --widen-factor 4 \
#                 --name WideResNet-40-4 \
#                 --typer ReLU \
#                 --retrain 0 \
#                 --epochs 400 \
#                 --save-dir runs_eval1


# python train.py --dataset cifar100 \
#                 --layers 40 \
#                 --widen-factor 4 \
#                 --name WideResNet-40-4 \
#                 --typer ReLU \
#                 --retrain 0 \
#                 --epochs 400 \
#                 --save-dir runs_f



# ## SVHN, ReLU
python train.py --dataset SVHN \
                --layers 16 \
                --widen-factor 4 \
                --name WideResNet-16-4 \
                --typer ReLU \
                --lr=0.01 \
                --retrain 0 \
                --warm_up \
                --droprate 0.0 \
                --epochs 1000 \
                --save-dir runs_svhn


# # SVHN, ReLU testing
python train.py --dataset SVHN -e \
                --layers 16 \
                --widen-factor 4 \
		        --prune="./runs_svhn/WideResNet-16-4/ReLU/checkpoint.pth.tar" \
                --name WideResNet-16-4 \
                --typer ReLU \
                --retrain 0 \
                --warm_up \
                --droprate 0.0 \
                --save-dir runs_svhn_evals

