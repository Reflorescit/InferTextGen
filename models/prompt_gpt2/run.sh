CUDA_VISIBLE_DEVICES=3 \
python main.py \
--name prefix100+none \
--method soft \
--soft_len 100 \
--do_train \
--do_pred \
--prompt_load_path ./prompt_encoders/none_gpt2-xl_soft_PromptEncoder.pkl \
--prompt_save_name none_gpt2-xl_soft_PromptEncoder.pkl \
--output_file_name none_soft100_gen.jsonl \
--n_shot 19200 \
--lr 5e-5 \
--early_stop 10 \
--log
