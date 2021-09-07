python oscar/run_captioning.py \
--do_eval \
--data_dir=../datasets/coco_caption \
--per_gpu_eval_batch_size=56 \
--num_workers=6 \
--num_beams=5 \
--max_gen_length=20 \
--eval_model_dir=../output/scst/checkpoint-31-896000
