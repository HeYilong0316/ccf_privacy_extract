export MAX_LENGTH=512
export BERT_MODEL=../user_data/output/output_layer_lstm_crf/0_fold



export DATA_DIR=../user_data/data/k_fold_10_510/fold_0
export OUTPUT_DIR=../user_data/output/output_layer_lstm_crf/0_fold
export TEST_DATA_DIR=../user_data/data/k_fold_10_510/
export BATCH_SIZE=32
export NUM_EPOCHS=$3
export WARM_UP=0.15

export SEED=666

export LEARNING_RATE=1e-05
export LW_LEARNING_RATE=1e-04
export WE_LR=1e-04
export WD_LR=1e-03
export LSTM_LEARNING_RATE=1e-03
export IDCNN_LEARNING_RATE=1e-03
export LINEAR_LEARNING_RATE=1e-03
export CRF_LEARNING_RATE=1e-02

export DROPOUT=0.1
export MAX_GRAD_NORM=1.0

export WEIGHT_DECAY=1e-02
export LW_WEIGHT_DECAY=1e-02
export WE_WEIGHT_DECAY=1e-02
export WD_WEIGHT_DECAY=1e-02
export LSTM_WEIGHT_DECAY=1e-02
export IDCNN_WEIGHT_DECAY=1e-02
export LINEAR_WEIGHT_DECAY=1e-02
export CRF_WEIGHT_DECAY=1e-02

export MAX_EPOCH=$4


echo "文件路径: " $DATA_DIR
echo "测试文件路径": $TEST_DATA_DIR
echo "BATCH_SIZE:" $BATCH_SIZE
echo "NUM_EPOCHS: " $NUM_EPOCHS
echo "MAX_EPOCH: " $MAX_EPOCH
echo "输出路径: " $OUTPUT_DIR

python -u run_ner.py \
--task_type NER \
--use_crf \
--word_vocab_path "../user_data/data/dicts/simple_tencent_vocab.txt" \
--word_embedding_path "../user_data/data/dicts/simple_tencent_embedding.npy" \
--data_dir $DATA_DIR \
--test_data_dir $TEST_DATA_DIR \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--seed $SEED \
--do_eval \
--do_predict \
--multi_layer_fusion \
--use_lstm \
# --disable_tqdm \



# --use_words \

# 






# --do_predict \

