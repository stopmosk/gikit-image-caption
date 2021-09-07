python oscar/run_captioning.py \
--do_eval \
--data_dir=../datasets/coco_my_oscar_vvl_tags_nms2 \
--per_gpu_eval_batch_size=48 \
--num_workers=6 \
--num_beams=5 \
--max_gen_length=20 \
--eval_model_dir=../output/big_base_xe_tmp2/checkpoint-13-256000

#--eval_model_dir=../pretrained_models/image_captioning/coco_captioning_bse_xe/checkpoint-60-66360
#--data_dir=/mnt/Toshiba2TB/big_vinvl_oscar \