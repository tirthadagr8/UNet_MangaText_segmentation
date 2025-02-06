from UNet import UNet, UNetConfig
from utility import *
from config import Config
from PIL import Image
from trainer import get_trainer
from dataset import *

config=Config()
unet_config=UNetConfig(in_channels=config.in_channels,out_channels=config.out_channels,img_size=config.img_size)
# images, annotations, labels=load_data()
# dataset=DatasetLoaderPrivate2(images, labels, annotations, config)
# model=UNet(unet_config)
model=UNet.from_pretrained(config.model_checkpoint)
path="C:/Users/tirth/Desktop/manga_text_ocr-main/mangaset9/008/page_71.png"

# trainer=get_trainer(config,model,dataset)
# trainer.train()
# model.save_pretrained(config.model_checkpoint)
inference(model,path,config.img_size,'cuda')
