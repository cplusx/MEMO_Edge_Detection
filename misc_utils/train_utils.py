import torch
from omegaconf import OmegaConf
import os
from pytorch_lightning.loggers import WandbLogger
from misc_utils.model_utils import instantiate_from_config, get_obj_from_str

def get_vae_model(args):
    if hasattr(args.trainer_args, 'precision'):
        torch_dtype = torch.float16 if args.trainer_args.precision == 16 else torch.float32
    else:
        torch_dtype = torch.float32
    if hasattr(args.vae.params, 'pretrained_model_name_or_path'):
        vae = get_obj_from_str(args.vae.target).from_pretrained(**args.vae.params, torch_dtype=torch_dtype)
    elif hasattr(args.vae.params, 'config'):
        vae = get_obj_from_str(args.vae.target).from_config(**args.vae.params)
    else:
        vae = get_obj_from_str(args.vae.target)(**args.vae.params)
    return vae

def get_vae_trainer(vae, vae_trainer_args):
    vae_trainer_class = vae_trainer_args['target']
    vae_trainer_args = vae_trainer_args['params']
    vae_trainer = get_obj_from_str(vae_trainer_class)(
        vae,
        **vae_trainer_args
    )
    return vae_trainer

def get_edge_unet_model(args):
    if hasattr(args.trainer_args, 'precision'):
        torch_dtype = torch.float16 if args.trainer_args.precision == 16 else torch.float32
    else:
        torch_dtype = torch.float32
    if hasattr(args.unet.params, 'pretrained_model_name_or_path'):
        unet = get_obj_from_str(args.unet.target).from_pretrained(**args.unet.params, torch_dtype=torch_dtype)
    elif hasattr(args.unet.params, 'config'):
        unet = get_obj_from_str(args.unet.target).from_config(**args.unet.params)
    else:
        unet = get_obj_from_str(args.unet.target)(**args.unet.params)
    return unet

def get_edge_unet_trainer(unet, unet_trainer_args):
    unet_trainer_class = unet_trainer_args['target']
    unet_trainer_args = unet_trainer_args['params']
    unet_trainer = get_obj_from_str(unet_trainer_class)(
        unet,
        **unet_trainer_args
    )
    return unet_trainer

def get_models(args):
    if hasattr(args.trainer_args, 'precision'):
        torch_dtype = torch.float16 if args.trainer_args.precision == 16 else torch.float32
    else:
        torch_dtype = torch.float32
    submodule_dict = {}
    if hasattr(args, 'vae'):
        print('INFO: loading vae from config')
        submodule_dict['vae'] = get_obj_from_str(args.vae.target).from_pretrained(**args.vae.params, torch_dtype=torch_dtype)
    if hasattr(args, 'scheduler'):
        print('INFO: loading scheduler from config')
        submodule_dict['scheduler'] = get_obj_from_str(args.scheduler.target).from_pretrained(**args.scheduler.params)
    if hasattr(args, 'denoiser'):
        print('INFO: loading unet from config')
        submodule_dict['denoiser'] = get_obj_from_str(args.denoiser.target)(**args.denoiser.params)
    if hasattr(args, 'cond_encoder'):
        print('INFO: loading cond_encoder from config')
        submodule_dict['cond_encoder'] = get_obj_from_str(args.cond_encoder.target)(**args.cond_encoder.params)
    pipe_class = get_obj_from_str(args.pipe.target)
    pipe = pipe_class(**args.pipe.params, **submodule_dict)
    return pipe

def get_edge_trainer(pipe, edge_model_configs):
    model_class = edge_model_configs['target']
    args = edge_model_configs['params']
    edge_trainer_class_instance = get_obj_from_str(model_class)
    edge_trainer = edge_trainer_class_instance(
        pipe,
        **args
    )
    return edge_trainer

def get_logger(args):
    wandb_logger = WandbLogger(
        project=args["expt_name"],
    )
    return wandb_logger

def get_callbacks(args, wandb_logger):
    callbacks = []
    for callback in args['callbacks']:
        if callback.get('require_wandb', False):
            # we need to pass wandb logger to the callback
            callback_obj = get_obj_from_str(callback.target)
            callbacks.append(
                callback_obj(wandb_logger=wandb_logger, **callback.params)
            )
        else:
            callbacks.append(
                instantiate_from_config(callback)
            )
    return callbacks

def get_dataset(args):
    from torch.utils.data import DataLoader
    data_args = args['data']
    train_shuffle = data_args.get('train_shuffle', True)
    val_shuffle = data_args.get('val_shuffle', False)
    train_set = instantiate_from_config(data_args['train'])
    val_set = instantiate_from_config(data_args['val'])
    if hasattr(data_args, 'collate_fn'):
        print('INFO: using custom collate_fn {}'.format(data_args.collate_fn.target))
        collate_fn = get_obj_from_str(data_args.collate_fn.target)
    else:
        collate_fn = None
    train_loader = DataLoader(
        train_set, batch_size=data_args['batch_size'], shuffle=train_shuffle,
        num_workers=4*len(args['trainer_args']['devices']), pin_memory=False,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=data_args['val_batch_size'],
        num_workers=len(args['trainer_args']['devices']), pin_memory=False,
        collate_fn=collate_fn, shuffle=val_shuffle
    )
    return train_loader, val_loader, train_set, val_set

def unit_test_create_dataset(config_path, split='train'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conf = OmegaConf.load(config_path)
    train_loader, val_loader, train_set, val_set = get_dataset(conf)
    if split == 'train':
        batch = next(iter(train_loader))
    else:
        batch = next(iter(val_loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

def unit_test_create_edge_trainer_model(config_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conf = OmegaConf.load(config_path)
    pipe = get_models(conf)
    edge_trainer = get_edge_trainer(pipe, conf['edge_trainer'])
    edge_trainer = edge_trainer.to(device)
    return edge_trainer

def unit_test_edge_trainer_training_step(config_path):
    edge_trainer = unit_test_create_edge_trainer_model(config_path)
    batch = unit_test_create_dataset(config_path)
    res = edge_trainer.training_step(batch, 0)
    return res

def unit_test_edge_trainer_val_step(config_path):
    edge_trainer = unit_test_create_edge_trainer_model(config_path)
    batch = unit_test_create_dataset(config_path, split='val')
    res = edge_trainer.validation_step(batch, 0)
    return res

def unit_test_create_vae(config_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conf = OmegaConf.load(config_path)
    vae = get_vae_model(conf)
    vae_trainer = get_vae_trainer(vae, conf['vae_trainer'])
    vae_trainer = vae_trainer.to(device)
    return vae_trainer

def unit_test_vae_training_step(config_path):
    vae = unit_test_create_vae(config_path)
    batch = unit_test_create_dataset(config_path)
    res = vae.training_step(batch, 0)
    return res

def unit_test_vae_val_step(config_path):
    vae = unit_test_create_vae(config_path)
    batch = unit_test_create_dataset(config_path, split='val')
    res = vae.validation_step(batch, 0)
    return res

def unit_test_create_edge_unet(config_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conf = OmegaConf.load(config_path)
    edge_unet = get_edge_unet_model(conf)
    unet_trainer = get_edge_unet_trainer(edge_unet, conf['unet_trainer'])
    unet_trainer = unet_trainer.to(device)
    return unet_trainer

def unit_test_unet_training_step(config_path):
    unet = unit_test_create_edge_unet(config_path)
    batch = unit_test_create_dataset(config_path)
    res = unet.training_step(batch, 0)
    return res

def unit_test_unet_val_step(config_path):
    unet = unit_test_create_edge_unet(config_path)
    batch = unit_test_create_dataset(config_path, split='val')
    res = unet.validation_step(batch, 0)
    return res