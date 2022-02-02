#~/bin/bash

err_report() {
	echo "An Error Occured on line $1"
	exit 1
}

trap 'err_report $LINENO' ERR

for i in `seq 1 30`; 
do
	echo $i
	MODEL_PATH="/path/to/hyperparm/models/$i/best_model.pt"
	EVAL_OUT="/path/to/output/Hypertune_Models/$i/eval_out/"
	DATASET_PATH="/path/to/Testing_Patch_Set/images"

	python predict.py $EVAL_OUT --checkpoint-path $MODEL_PATH --dataset_path $DATASET_PATH --patch_set

	dddlog "Model Complete" "Model $i/30"	
	
done;

