python oscar/run_captioning.py \
--do_eval \
--data_dir=../datasets/coco/coco_my_oscar_vvl \
--per_gpu_eval_batch_size=120 \
--num_workers=6 \
--num_beams=1 \
--max_gen_length=20 \
--eval_model_dir=../pretrained_models/image_captioning/coco_captioning_base_xe/checkpoint-60-66360 \
--fp16
