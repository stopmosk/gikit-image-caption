python oscar/run_captioning.py \
--do_test \
--data_dir=../datasets/coco_caption \
--per_gpu_eval_batch_size=1 \
--num_workers=1 \
--num_beams=5 \
--max_gen_length=20 \
--eval_model_dir=../pretrained_models/image_captioning/coco_captioning_base_scst/checkpoint-15-66405
