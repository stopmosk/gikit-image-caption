python tools/run.py \
--tasks captioning \
--datasets m4c_textcaps \
--model cnmt \
--config configs/cnmt_rt.yml \
--save_dir save/eval_my/ \
--run_type inference \
--evalai_inference 1 \
--resume_file ../best.ckpt

#--resume_file save/cnmt/m4c_textcaps_cnmt/best.ckpt