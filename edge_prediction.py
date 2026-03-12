import os
import shutil
import numpy as np
import torch
import argparse
import cv2
from glob import glob
from tqdm import tqdm
from misc_utils.train_utils import get_obj_from_str
from misc_utils.train_utils import get_models, get_edge_trainer
from omegaconf import OmegaConf
def pad_image_to_fit_model(image, unit_size=16):
    h, w = image.shape[:2]
    if h % unit_size == 0:
        h_pad = 0
    else:
        h_pad = unit_size - h % unit_size
    if w % unit_size == 0:
        w_pad = 0
    else:
        w_pad = unit_size - w % unit_size
    image = np.pad(image, ((0, h_pad), (0, w_pad), (0, 0)), mode='symmetric')
    return image, h_pad, w_pad

def predict_one_image(pipe, image_path, guidance_scale=2.5, inference_steps=50, dino_size_mode='fixed', conf_thres=0.5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, h_pad, w_pad = pad_image_to_fit_model(image, unit_size=32)
    image = image / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)

    if dino_size_mode == 'fixed':
        dino_size = (224, 224)
    elif dino_size_mode == 'adaptive':
        H, W = image.shape[2], image.shape[3]
        dino_size = ((H // 14) * 14, (W // 14) * 14)
        print(f"image size: ({H}, {W}), dino size: {dino_size}")

    with torch.no_grad(), torch.amp.autocast(device, dtype=torch.float16, enabled=True):
        import inspect
        sig = inspect.signature(pipe.__call__)
        if 'max_inference_steps' in sig.parameters:
            pred_dict = pipe(
                images=image,
                max_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                dino_additional_kwargs={'dino_size': dino_size},
                conf_thres=conf_thres,
            )
        else:
            pred_dict = pipe(
                images=image,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                dino_additional_kwargs={'dino_size': dino_size},
                conf_thres=conf_thres,
            )
        pred = pred_dict['edges']  # (B, H, W)
        pred_probs = pred_dict['pred_probs']  # (B, H, W, C)
    pred = pred[:, :image.shape[2]-h_pad, :image.shape[3]-w_pad]
    pred_probs = pred_probs[:, :image.shape[2]-h_pad, :image.shape[3]-w_pad, :]  # (B, H, W, C)
    pred = pred[0] # remove batch dimension
    pred_probs = pred_probs[0]  # (H, W, C)
    return pred.astype(np.int16), pred_probs.astype(np.float32)  # (H, W, C)

def multiclass_to_prediction(pred_prob):
    # pred_prob: (num_classes, H, W) num class inlcude a background
    max_cls = pred_prob.shape[0] - 1
    category_to_binary_map = np.linspace(0, 1, max_cls+1, endpoint=True)
    category_to_binary_map = category_to_binary_map.reshape(-1, 1, 1)  # (num_classes+1, 1, 1)

    prediction = (pred_prob * category_to_binary_map).sum(axis=0)  # (H, W)
    return (prediction * 255).astype(np.uint8)  # (H, W)


def quantized_to_binarized(pred_label):
    return (pred_label > 0).astype(np.uint8) * 255


def archive_legacy_output_dirs(save_folder):
    legacy_root = os.path.join(save_folder, '_legacy_outputs')
    for legacy_name in ('quantized', 'colorized'):
        source_dir = os.path.join(save_folder, legacy_name)
        if not os.path.isdir(source_dir):
            continue

        target_dir = os.path.join(legacy_root, legacy_name)
        suffix = 1
        while os.path.exists(target_dir):
            target_dir = os.path.join(legacy_root, f'{legacy_name}_{suffix}')
            suffix += 1

        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        shutil.move(source_dir, target_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_folder', default="test_images", type=str, help='the folder of test images')
    parser.add_argument('--save_folder', default="results", type=str, help='the folder to save results')
    parser.add_argument('--config_file', default="configs/edge_pred_base/quant_6/edge_unet_test.yaml", type=str, help='the default config file')
    parser.add_argument('--model_path', type=str, help='the path of the model')
    parser.add_argument('--base_model_path', type=str, default=None, help='optional base pretrained checkpoint for LoRA finetuned models')
    parser.add_argument('--guidance_scale', type=float, help='the guidance scale for the MEMO model', default=2.5)
    parser.add_argument('--max_steps', type=int, help='the max steps for the MEMO model', default=50)
    parser.add_argument('--dino_size_mode', type=str, help='the dino image size mode, fixed (224,224) or adaptive (nearest to 14xN of image size)', default='fixed')
    parser.add_argument('--conf_thres', type=float, help='the confidence threshold to keep during the denoise process', default=0.5)
    args = parser.parse_args()

    guidance_scale = args.guidance_scale

    image_paths = sorted(glob(os.path.join(args.test_folder, '**/*.jpg'), recursive=True) + 
                        glob(os.path.join(args.test_folder, '**/*.png'), recursive=True))

    model_path = args.model_path
    save_folder = os.path.join(args.save_folder)
    archive_legacy_output_dirs(save_folder)
    save_folder_prediction = os.path.join(save_folder, 'prediction')
    save_folder_binarized = os.path.join(save_folder, 'binarized')
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_folder_prediction, exist_ok=True)
    os.makedirs(save_folder_binarized, exist_ok=True)

    config_file = args.config_file
    config = OmegaConf.load(config_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer_target = config.edge_trainer.target
    if 'LoRA' in trainer_target:
        if args.base_model_path is not None:
            config.edge_trainer.params.init_weights = args.base_model_path
        models = get_models(config)
        edge_trainer = get_edge_trainer(models, edge_model_configs=config.edge_trainer)
        denoiser = edge_trainer.denoiser
    else:
        denoiser = get_obj_from_str(config.denoiser.target)(**config.denoiser.params)
    model_weights = torch.load(model_path, map_location='cpu')
    if 'module' in model_weights:
        model_weights = model_weights['module']
    ema_model_weights = {k.replace('ema_denoiser.module.', ''): v for k, v in model_weights.items() if 'ema_denoiser.module.' in k}
    # ema_model_weights = {k.replace('denoiser.', ''): v for k, v in model_weights.items() if 'denoiser.' in k and 'ema' not in k}
    denoiser.load_state_dict(ema_model_weights)

    pipe = get_obj_from_str(config.pipe.target)(denoiser=denoiser, **config.pipe.params)
    pipe = pipe.to(device)

    for image_path in tqdm(image_paths):
        image_name = os.path.basename(image_path)
        save_path_prediction = os.path.join(save_folder_prediction, image_name.replace('.jpg', '.png'))
        save_path_binarized = os.path.join(save_folder_binarized, image_name.replace('.jpg', '.png'))
        if os.path.exists(save_path_prediction):
            continue

        pred_label, pred_prob = predict_one_image(pipe, image_path, guidance_scale=guidance_scale, inference_steps=args.max_steps, dino_size_mode=args.dino_size_mode, conf_thres=args.conf_thres)
        prediction = multiclass_to_prediction(pred_prob.transpose(2, 0, 1))
        binarized = quantized_to_binarized(pred_label)
        cv2.imwrite(save_path_prediction, prediction)
        cv2.imwrite(save_path_binarized, binarized)
        # break