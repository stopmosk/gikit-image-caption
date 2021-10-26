python -m torch.distributed.launch --nproc_per_node=8 \
oscar/run_captioning.py \
--do_eval \
--data_dir=../coco_preexacted \
--per_gpu_eval_batch_size=512 \
--num_workers=0 \
--num_beams=5 \
--max_gen_length=20 \
--eval_model_dir=../pretrained_models/image_captioning/coco_captioning_large_xe/checkpoint-25-28000

#--eval_model_dir=../pretrained_models/image_captioning/coco_captioning_bse_xe/checkpoint-60-66360
#--data_dir=/mnt/Toshiba2TB/big_vinvl_oscar \