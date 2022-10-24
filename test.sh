export CLASSPATH="/data/tianmu/data/PreSumm/stanford-corenlp-4.5.1/stanford-corenlp-4.5.1.jar"
RAW_PATH="/data/tianmu/data/PreSumm/raw_stories"
TOKENIZED_PATH="/data/tianmu/data/PreSumm/merged_stories_tokenized"
mkdir -p $TOKENIZED_PATH
LOG_FILE="/data/tianmu/data/PreSumm/token.log"
python src/preprocess.py -mode tokenize -raw_path $RAW_PATH -save_path $TOKENIZED_PATH -log_file $LOG_FILE


RAW_PATH="/data/tianmu/data/PreSumm/merged_stories_tokenized"
JSON_PATH="/data/tianmu/data/PreSumm/story_json"
MAP_PATH="/data/tianmu/data/PreSumm/story_map"
LOG_FILE="/data/tianmu/data/PreSumm/lines_token.log"
mkdir -p $JSON_PATH
mkdir -p $MAP_PATH

python src/preprocess.py -mode format_to_lines -raw_path $RAW_PATH -save_path $JSON_PATH -n_cpus 1 -use_bert_basic_tokenizer false -map_path $MAP_PATH -log_file $LOG_FILE

JSON_PATH="/data/tianmu/data/PreSumm/story_json"
BERT_DATA_PATH="/data/tianmu/data/PreSumm/story_bert"
LOG_FILE="/data/tianmu/data/PreSumm/bert_token.log"

mkdir -p $BERT_DATA_PATH
python src/preprocess.py -mode format_to_bert -raw_path $JSON_PATH -save_path $BERT_DATA_PATH  -lower -n_cpus 1 -log_file $LOG_FILE


BERT_DATA_PATH="/data/tianmu/data/PreSumm/story_bert"
MODEL_PATH="/data/tianmu/checkpoints/PreSumm/story"
LOG_FILE="/data/tianmu/checkpoints/PreSumm/story/train.log"

CUDA_VISIBLE_DEVICES=1 python src/train.py -task ext \
-mode train -bert_data_path $BERT_DATA_PATH \
-ext_dropout 0.1 -model_path $MODEL_PATH -lr 2e-3 -visible_gpus 0 \
-report_every 50 \
-save_checkpoint_steps 100 -batch_size 3000 \
-train_steps 50000 \
-accum_count 2 \
-log_file $LOG_FILE \
-use_interval true \
-warmup_steps 10000 \
-max_pos 512

# test and inference (generate pred output)

BERT_DATA_PATH="/data/tianmu/data/PreSumm/story_bert"
MODEL_PATH="/data/tianmu/checkpoints/PreSumm/story"
LOG_FILE="/data/tianmu/checkpoints/PreSumm/story/test.log"
MODEL_CHECKPOINT="/data/tianmu/checkpoints/PreSumm/story/model_step_2600.pt"
RESULT_PATH="/data/tianmu/checkpoints/PreSumm/story/result"
TEMP_DIR="/data/tianmu/checkpoints/PreSumm/story/temp"
TEST_PARTITION="test"
CUDA_VISIBLE_DEVICES=1 python src/train.py -task ext \
-mode test -bert_data_path $BERT_DATA_PATH \
-test_from $MODEL_CHECKPOINT \
-test_partition $TEST_PARTITION \
-result_path $RESULT_PATH \
-ext_dropout 0.1 -model_path $MODEL_PATH -lr 2e-3 -visible_gpus 0 \
-report_every 50 \
-save_checkpoint_steps 50 \
-test_batch_size 3000 \
-train_steps 50000 \
-accum_count 2 \
-log_file $LOG_FILE \
-use_interval true \
-block_trigram true \
-recall_eval true \
-report_rouge false \
-temp_dir $TEMP_DIR \
-warmup_steps 10000 \
-max_pos 512








# python train.py -task abs -mode validate -batch_size 3000 \
# -test_batch_size 500 -bert_data_path BERT_DATA_PATH \
# -log_file ../logs/val_abs_bert_cnndm -model_path MODEL_PATH -sep_optim true \
# -use_interval true -visible_gpus 1 -max_pos 512 -min_length 20 \
# -max_length 100 -alpha 0.9 -result_path ../logs/abs_bert_cnndm 