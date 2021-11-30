#!/bin/bash

# Oscar+ evaluation on custom checkpoint

python oscar/run_captioning.py \
--do_test \
--test_yaml=all.yaml \
--data_dir=../datasets/coco_my_ocr_13_vvl \
--per_gpu_eval_batch_size=32 \
--num_workers=4 \
--num_beams=1 \
--max_gen_length=20 \
--output_dir=../output/tmp_results \
--eval_model_dir=../output/txtcps_cleaned_base_xe_pos/checkpoint-77-111000

#--eval_model_dir=../output/agg_coco_textcaps_cleaned_1/checkpoint-4-390000
#--eval_model_dir=../output/txtcps_cleaned_base_xe1/checkpoint-50-73000


# python oscar/run_captioning.py \
# --do_eval \
# --data_dir=../datasets/coco/coco_oscar_preexacted_vinvl \
# --per_gpu_eval_batch_size=32 \
# --num_workers=4 \
# --num_beams=1 \
# --max_gen_length=20 \
# --output_dir=../output/tmp_results \
# --eval_model_dir=../output/coco_my_oscar_vvl_!/checkpoint-1-10000
