#!/bin/bash

TEMPLATES_FOLDER="../new-templates-batch1"
RESULT_MODELS_FOLDER="ce-models"
TEST_IMAGE="../Space_Imgs/840x560/img12.JPG"

PAR_WEIGHT=1.5e1
PAR_ITERATIONS=1000
PAR_BATCH_SIZE=20

for f in `find $TEMPLATES_FOLDER -name "*.jpg" -o -name "*.png" -o -name "*.JPG"`; do 
	fname=$(basename "$f"); 
	dname="${fname%.*}"; 
	MODEL_FOLDER=$RESULT_MODELS_FOLDER/$dname
	TRAIN_TEST_FOLDER=$MODEL_FOLDER/test-out
	mkdir -p $MODEL_FOLDER; 
	mkdir -p $TRAIN_TEST_FOLDER;

	echo "training for $f results will be in $MODEL_FOLDER"; 
	python style.py --style $f \
		--checkpoint-dir $MODEL_FOLDER \
		--test $TEST_IMAGE \
		--test-dir $TRAIN_TEST_FOLDER 
		--content-weight $PAR_WEIGHT \
		--checkpoint-iterations $PAR_ITERATIONS \
		--batch-size $PAR_BATCH_SIZE;
done;

