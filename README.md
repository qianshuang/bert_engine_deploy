python3 run_classifier.py \
--data_dir=./data \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--task_name=comm100 \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--output_dir=output \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--do_train=True \
--do_eval=True \
--num_train_epochs=1