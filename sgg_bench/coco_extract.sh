# python tools/test_sg_net.py --config-file models/vinvl/o365/vinvl_vg_oscar.yaml
# python tools/test_sg_net.py --config-file models/vinvl/vg/vinvl_textcaps.yaml
python tools/test_sg_net.py --config-file models/vinvl/vg/vinvl_big.yaml DATA_DIR "../../datasets/big" OUTPUT_DIR "../../datasets/big_vinvl" DATALOADER.NUM_WORKERS 8 TEST.IMS_PER_BATCH 8
# DATA_DIR "/mnt/Toshiba2TB/dataset_hic_encoded" OUTPUT_DIR "/mnt/Toshiba2TB/big_vinvl"
