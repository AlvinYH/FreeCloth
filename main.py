import yaml
import glob
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from lib.freecloth_model import FreeClothModel


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(42, workers=True)
    torch.set_num_threads(10) 

    # setup datamodule
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    if cfg.mode == 'train':  # train
        datamodule.setup(stage='fit')

        # setup logger
        with open('.hydra/config.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        logger = pl.loggers.WandbLogger(project='snarf', config=config)

        # load model
        model = FreeClothModel(cfg=cfg)
        if cfg.resume:
            if cfg.epoch == 'last':
                checkpoint_path = './checkpoints/last.ckpt'
            else:
                checkpoint_path = sorted(glob.glob('./checkpoints/epoch=%d*.ckpt'%cfg.epoch))[0]
        else:
            checkpoint_path = None
        
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=-1,
            monitor=None, 
            save_last=True,
            dirpath=cfg.ckpt_dir,
            every_n_train_steps=cfg.save_ckpt_train_steps
        )

        trainer = pl.Trainer(accelerator='gpu', logger=logger, callbacks=[checkpoint_callback], **cfg.trainer)
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)


    else: # test
        datamodule.setup(stage='test')

        trainer = pl.Trainer(accelerator='gpu', **cfg.trainer)

        if cfg.epoch == 'last':
            checkpoint_path = './checkpoints/last.ckpt'
        else:
            checkpoint_path = glob.glob('./checkpoints/epoch=%d*.ckpt'%cfg.epoch)[0]

        model = FreeClothModel.load_from_checkpoint(checkpoint_path=checkpoint_path, cfg=cfg)
        trainer.test(model, datamodule=datamodule, verbose=True)

    
if __name__ == '__main__':
    main()