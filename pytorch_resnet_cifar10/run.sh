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
