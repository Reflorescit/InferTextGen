CUDA_VISIBLE_DEVICES=5 \
python main.py \
--name prefix100_none \
--method prefixTuning \
--soft_len 100 \
--do_train \
--do_pred \
--prompt_load_path ./prompt_encoders/none_prefix100_prompt.pkl \
--prompt_save_name none_prefix100_prompt.pkl \
--output_file_name none_preifx100_gen.jsonl \
--n_shot 19200 \
--lr 0.3 \
--early_stop 10 \
--log
