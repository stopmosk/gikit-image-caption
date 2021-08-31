python tools/test_sg_net.py --config-file models/vinvl/vg/huawei_1000.yaml
# python tools/test_sg_net.py --config-file models/vinvl/vg/vinvl_big.yaml DATA_DIR "../../datasets/big" OUTPUT_DIR "../../datasets/big_vinvl" DATALOADER.NUM_WORKERS 8 TEST.IMS_PER_BATCH 8
# python tools/test_sg_net.py --config-file models/vinvl/vg/vinvl_big.yaml DATA_DIR "/mnt/Toshiba2TB/dataset_hic_encoded" OUTPUT_DIR "/mnt/Toshiba2TB/big_vinvl" DATALOADER.NUM_WORKERS 6 TEST.IMS_PER_BATCH 2 MODEL.ROI_HEADS.NMS_FILTER 2
