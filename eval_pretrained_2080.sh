python oscar/run_captioning.py \
--do_test \
--data_dir=../datasets/textcaps_vvl \
--per_gpu_eval_batch_size=56 \
--num_workers=6 \
--num_beams=5 \
--max_gen_length=20 \
--eval_model_dir=../pretrained_models/image_captioning/coco_captioning_base_xe/checkpoint-60-66360
