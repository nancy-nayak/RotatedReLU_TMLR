
# Install the required packages by running the following command (stay inside the directory codes_ICCV2023)
$ conda env create -f rrelujan.yml
# and activate them
$ conda activate rrelujan

# Instructions to obtain the results in Table 1. For ResNets change your directory to pytorch_resnet_cifar10.
# Copy-paste the corresponding commands to the run.sh file inside the directory and run.
# ==================================================================================================================

# 1. To train the ResNet20 architecture with ReLU and CIFAR10 dataset:

for model in resnet20
do
	python -u trainer_adv_Jan.py \
						--arch=$model\
						--dataset=CIFAR10 \
						--epoch=1200 \
						--save-dir=resultsICCV/save_standard_$model
done

# 2. To train the ResNet20 architecture with ReLU and CIFAR100 dataset:

for model in resnet20_f
do
	python -u trainer_adv_Jan.py \
						--arch=$model\
						--dataset=CIFAR100 \
						--epoch=1200 \
						--save-dir=resultsICCV/save_standard_$model
done


# 3. To train the ResNet20 architecture with RReLU and CIFAR10 dataset:

for model in resnet20_rotatedrelu_maam
do
	python -u trainer_adv_Jan.py \
						--arch=$model\
						--dataset=CIFAR10 \
						--epoch=1200 \
						--save-dir=resultsICCV/save_rrelu_$model
done

# 4. To train the ResNet20 architecture with RReLU and CIFAR100 dataset:

for model in resnet20_rotatedrelu_maam_f
do
	python -u trainer_adv_Jan.py \
						--arch=$model\
						--dataset=CIFAR100 \
						--epoch=1200 \
						--save-dir=resultsICCV/save_rrelu_$model
done

# 5. To test the model and to find out how many filters are inactive and layerwise inactive filters for the 
# ResNet20 architecture with RReLU and CIFAR10 dataset:
# Before testing, make sure that you have the fully trained model saved inside the directory ./pytorch_resnet_cifar10/resultsICCV/save_rrelu_resnet20_rotatedrelu_maam
# By changing the value of zeta, you can have different level of pruning. 
# Find a zeta that has no/negligible degradation in performance using validation set i.e. validation=1.
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
# Then test using that zeta after using validation=0
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

# 6. From the list of filters alive along the depth, you can calculate the number of parameters and FLOPs by running the following.
# Enter the list of alive filters obtained from the above command to _listrrelu in count.py and count_preactivation.py
# ===========================================================================================================
# For ResNet20, ResNet56, WRN-40-4, WRN-16-4, ResNet-50 run
python count.py
# For ResNet-110-pre and ResNet-164-pre, run
python count_preactivation.py


# 7. The firststep in the two step training method for showing RReLU as coarse feature extractor
# (results in Table 3) 
# =====================================================================================================
for model in resnet20_rotatedrelu_maam_f
do
	python -u trainer_adv_Jan.py \
						--arch=$model\
						--dataset=CIFAR100 \
						--twostepfirststep=1 \
						--epoch=500 \
						--save-dir=resultsICCV/save_rrelu_twostepfirststep_$model
done

# 8. Test the same
for model in resnet20_rotatedrelu_maam
do
	python -u trainer_adv_Jan.py \
						--arch=$model -e \
						--prune="./resultsICCV/save_rrelu_twostepfirststep_resnet20_rotatedrelu_maam/checkpoint_best.th" \
						--dataset=CIFAR10 \
						--twostepfirststep=1 \
						--zeta=0.03 \
						--epoch=500 \
						--save-dir=resultsICCV/save_rrelu_twostepfirststep_eval_$model
done 

# 9. The secondstep in the two step training method for CIFAR10
for model in resnet20_rotatedrelu_maam
do
	python -u trainer_adv_Jan.py \
						--arch=$model\
						--dataset=CIFAR10 \
						--twostepsecondstep=1 \
						--zeta=0.03 \
						--epoch=700 \
						--save-dir=resultsICCV/save_rrelu_twostepsecondstep_$model
done

## 10. Figure 7.b is shown for WRN-40-4, so use the list of filters alive along the depth for list_1 and run 
# =======================================================================================================
python filterlength.py

# For WideResNet results in Table 1, change your directory to WideResNet-pytorch, create a run.sh file, copy-paste the
# corresponding commands and run:
# =====================================================================================================

# 11. To train the WRN-40-4 architecture with ReLU and CIFAR10 dataset 
# (for CIFAR100 dataset, replace cifar10 with cifar100 and runs with runs_f):

python train.py --dataset cifar10 \
                --layers 40 \
                --widen-factor 4 \
                --name WideResNet-40-4 \
                --typer ReLU \
                --retrain 0 \
                --epochs 400 \
                --save-dir runs

# 12. To test the same (for CIFAR100 dataset, replace cifar10 with cifar100, runs with runs_f and runs_eval with runs_eval_f):
python train.py --dataset cifar10 \
                --layers 40 -e \
                --prune="./runs/WideResNet-40-4/ReLU/model_best.pth.tar" \
                --widen-factor 4 \
                --name WideResNet-40-4 \
                --typer ReLU \
                --retrain 0 \
                --epochs 400 \
                --save-dir runs_eval

# 13. To train the WRN-16-4 architecture with ReLU and SVHN dataset 
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
# 14. To test the same:
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


# 15. To train the WRN-40-4 architecture with RReLU and CIFAR10 dataset 
python train.py --dataset cifar10 \
                --layers 40 \
                --widen-factor 4 \
                --name WideResNet-40-4 \
                --typer RReLU \
                --retrain 0 \
                --epochs 400 \
                --save-dir runs

# 16. To test the same:
python train.py --dataset cifar10 -e \
                --prune="./runs/WideResNet-40-4/RReLU/model_best.pth.tar" \
                --layers 40 \
                --widen-factor 4 \
                --name WideResNet-40-4 \
                --typer RReLU \
                --retrain 0 \
                --zeta=0.04 \
                --epochs 400 \
                --save-dir runs_eval

# 17. For SVHN and RReLU with WRN-16-4 architecture
python train.py --dataset SVHN \
                --layers 16 \
                --widen-factor 4 \
                --name WideResNet-16-4 \
                --typer RReLU \
                --lr=0.01 \
                --retrain 0 \
                --warm_up \
                --droprate 0.0 \
                --epochs 1000 \
                --save-dir runs_svhn  
# 18. To test the same
python train.py --dataset SVHN -e \
                --layers 16 \
                --prune="./runs_svhn/WideResNet-16-4/RReLU/model_best.pth.tar" \
                --widen-factor 4 \
                --name WideResNet-16-4 \
                --typer RReLU \
                --retrain 0 \
                --warm_up \
                --zeta=0.04 \
                --droprate 0.0 \
                --epochs 1000 \
                --save-dir runs_svhn_eval 


# 19. For MNIST results in Table 1 and RReLU with FCNN architecture stay inside codes_ICCV2023 directory and run
# ==============================================================================================================
python MNIST.py
# Once trained, for only testing, comment the lines for training as directed in the code



# 20. Figure 5.a and Figure 5.b which are our baselines are reproduced from the code provided by the authors. 
# Go to directory network-slimming and run appropriate code from run.sh


# The results in Table 4 are found by running the below commands from the directory pytorch_resnet_cifar10. Number of perturbation steps for calculating LLC
# is set inside utils/common.py as 60.
# =================================================================================================================
# 21. For ReLU,
for model in resnet20
do
	python -u trainer_adv_Jan.py \
						--arch=$model -e \
						--dataset=CIFAR10 \
						--prune="./resultsICCV/save_standard_resnet20/checkpoint_best.th" \
						--epoch=1200 \
						--epsfgsm=0.031 \
						--epspgd=0.031 \
						--advtest=1 \
						--save-dir=resultsICCV/save_standard_eval_$model
done
# 22. For RReLU,
for model in resnet20_rotatedrelu_maam
do
	python -u trainer_adv_Jan.py \
						--arch=$model -e \
						--dataset=CIFAR10 \
						--prune="./resultsICCV/save_rrelu_resnet20_rotatedrelu_maam/checkpoint_best.th" \
						--epsfgsm=0.031 \
						--epspgd=0.031 \
						--advtest=1 \
						--epoch=1200 \
						--save-dir=resultsICCV/save_rrelu_eval_$model
done

# 23. If you want to train multiple different instant of the same network, 
# uncomment torch.manual_seed(0) in line 28 of pytorch_resnet_cifar10/trainer_adv_Jan.py