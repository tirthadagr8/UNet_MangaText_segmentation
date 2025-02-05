from UNet import UNet
from utility import *
from config import Config
from PIL import Image
from trainer import get_trainer
from dataset import *

config=Config()
# images, annotations, labels=load_data()
# dataset=DatasetLoaderPrivate2(images, labels, annotations, config)
model=UNet(config=config)
path="C:/Users/tirth/Desktop/manga_text_ocr-main/mangaset9/008/page_71.png"

# trainer=get_trainer(config,model,dataset)
# trainer.train()
# torch.save(model.state_dict(),config.model_checkpoint)
inference(model,path,config.img_size,'cuda')
