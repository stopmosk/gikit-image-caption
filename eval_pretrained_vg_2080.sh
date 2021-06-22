python oscar/run_captioning.py \
--do_eval \
--data_dir=../datasets/coco/coco_oscar_preexacted_vg \
--per_gpu_eval_batch_size=54 \
--num_workers=6 \
--num_beams=5 \
--max_gen_length=20 \
--eval_model_dir=../pretrained_models/image_captioning/base-vg-labels/checkpoint-29-66420
