# MEMO_Edge_Detection
The code for Masked Edge Prediction Model

### Install the dependencies
```bash
pip install -r requirements.txt
cd opencv_edge
bash dld.sh
```

### Download SGED dataset
https://huggingface.co/datasets/cplusx/SGED


### Use SGED trained Model
https://huggingface.co/cplusx/MEMO_laion_binary/tree/main/checkpoint

```bash
python edge_prediction \
    --test_folder [PATH_TO_TEST_FOLDER] \
    --save_folder [PATH_TO_SAVE_FOLDER] \
    --config_file configs/discrete_BSDS_finetune/binary_lora_default.yaml \
    --model_path [PATH_TO_MODEL] \
    --guidance_scale 1.4 \
    --max_steps 20
```

### To train model on SGED
```bash
python train.py --config_file configs/binary/discrete_v2data_binary_dinov2.yaml
```

### To finetune on SGED pretrained model
```bash
python train.py --config_file configs/discrete_BIPED_finetune/binary_lora_default.yaml
```
There are several configurations to modify in the config file
`init_weights`: change to the SGED pretrained weights
`root_dir`: change to the root of the BIPEDv2 dataset, check the `edge_datasets.edge_datasets.BIPEDv2` for details.