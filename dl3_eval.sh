python oscar/run_captioning.py \
--do_eval \
--data_dir=../datasets/coco/coco_oscar_vvl_my \
--per_gpu_eval_batch_size=120 \
--num_workers=6 \
--num_beams=1 \
--max_gen_length=20 \
--eval_model_dir=../output/coco/checkpoint-25-76000 \
--fp16
