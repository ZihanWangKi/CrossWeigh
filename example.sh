export CONLL03_TRAIN_FILE=PATH_TO_CONLL03_TRAIN_FILE
export CONLL03_DEV_FILE=PATH_TO_CONLL03_DEV_FILE
export CONLL03_TEST_FILE=PATH_TO_CONLL03_TEST_FILE
export DATA_FOLDER_PREFIX=PATH_TO_DATA_FOLDER
export MODEL_FOLDER_PREFIX=PATH_TO_MODEL_FOLDER
export WEIGHED_MODEL_FOLDER_NAME=/weighed
mkdir ${DATA_FOLDER_PREFIX}/${WEIGHED_MODEL_FOLDER_NAME}

# creating splits
for splits in $(seq 1 1 3); do
    SPLIT_FOLDER=${DATA_FOLDER_PREFIX}/split-${splits}
    python split.py --input_files ${CONLL03_TRAIN_FILE} ${CONLL03_DEV_FILE} \
                    --output_folder ${SPLIT_FOLDER} \
                    --schema iob
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
                  --train_files CONLL03_TRAIN_FILE CONLL03_DEV_FILE \
                  --train_file_schema iob \
                  --output ${WEIGHED_MODEL_FOLDER}/${WEIGHED_MODEL_FOLDER_NAME}/train.bio

# train the final model
python flair_scripts/flair_ner.py --folder_name ${WEIGHED_MODEL_FOLDER_NAME} \
                                  --data_folder_prefix ${DATA_FOLDER_PREFIX} \
                                  --model_folder_prefix ${MODEL_FOLDER_PREFIX} \
                                  --include_weight