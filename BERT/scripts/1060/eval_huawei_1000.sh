python oscar/run_captioning.py \
--do_eval \
--data_dir=../datasets/huawei_1000 \
--per_gpu_eval_batch_size=24 \
--num_workers=4 \
--num_beams=5 \
--max_gen_length=20 \
--eval_model_dir=../models/checkpoint-13-256000 \
--test_yaml='val.yaml'

#--eval_model_dir=../pretrained_models/image_captioning/coco_captioning_bse_xe/checkpoint-60-66360
#--data_dir=/mnt/Toshiba2TB/big_vinvl_oscar \