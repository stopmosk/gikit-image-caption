python oscar/run_captioning.py \
--do_eval \
--test_yaml='all.yaml' \
--add_ocr_labels \
--data_dir=../datasets/coco_my_ocr_13_vvl \
--per_gpu_eval_batch_size=56 \
--num_workers=6 \
--num_beams=5 \
--max_gen_length=20 \
--eval_model_dir=../output/coco_my_vvl_ocr_xe_2/checkpoint-51-152000

#--eval_model_dir=../output/txtcps_xe_clro_posenc_roz_pos_xywh/checkpoint-79-182960
