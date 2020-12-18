export CONLL03_TRAIN_FILE=data/conllpp_train.txt
export CONLL03_DEV_FILE=data/conllpp_dev.txt
export CONLL03_TEST_FILE=data/conllpp_test.txt
export DATA_FOLDER_PREFIX=splitdata
export MODEL_FOLDER_PREFIX=model
export WEIGHED_MODEL_FOLDER_NAME=weighed
mkdir -p ${DATA_FOLDER_PREFIX}/${WEIGHED_MODEL_FOLDER_NAME}

# creating splits
for splits in $(seq 1 1 3); do
    SPLIT_FOLDER=${DATA_FOLDER_PREFIX}/split-${splits}
    python split.py --input_files ${CONLL03_TRAIN_FILE} ${CONLL03_DEV_FILE} \
                    --output_folder ${SPLIT_FOLDER} \
                    --schema iob \
		    --folds 10
done

# training each split/fold
for splits in $(seq 1 1 3); do
    for folds in $(seq 0 1 9); do
        FOLD_FOLDER=split-${splits}/fold-${folds}
        python flair_scripts/flair_ner.py --folder_name ${FOLD_FOLDER} \
                                          --data_folder_prefix ${DATA_FOLDER_PREFIX} \
                                          --model_folder_prefix ${MODEL_FOLDER_PREFIX}
    done
done

# collecting results and forming a weighted train set.
python collect.py --split_folders ${DATA_FOLDER_PREFIX}/split-*  \
                  --train_files $CONLL03_TRAIN_FILE $CONLL03_DEV_FILE \
                  --train_file_schema iob \
                  --output ${DATA_FOLDER_PREFIX}/${WEIGHED_MODEL_FOLDER_NAME}/train.bio

# train the final model
python flair_scripts/flair_ner.py --folder_name ${WEIGHED_MODEL_FOLDER_NAME} \
                                  --data_folder_prefix ${DATA_FOLDER_PREFIX} \
                                  --model_folder_prefix ${MODEL_FOLDER_PREFIX} \
                                  --include_weight
