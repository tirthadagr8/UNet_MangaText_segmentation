class Config:
    def __init__(self):
        self.encoder_model_name='google/vit-base-patch16-224-in21k'
        self.decoder_model_name='dis|tilgpt2'
        self.feature_extractor_name='microsoft/resnet-50'
        self.tokenizer_name='tirthadagr8/CustomOCR'
        self.in_channels=3
        self.out_channels=1
        self.img_size=1024
        self.device='cuda'
        self.model_checkpoint='C:/Users/tirth/Desktop/UNet_MangaText_segmentation/model_checkpoint/'
        self.NEPTUNE_API_TOKEN=""
        self.NEPTUNE_PROJECT=""
        self.report_to='none' # 'neptune' for logging to Neptune
        self.trainer_config={
            'output_dir':"./ocr-model",
            'learning_rate':5e-3,
            'per_device_train_batch_size':2,
            'lr_scheduler_type':'cosine',
            'num_train_epochs':2,
            'logging_dir':"./logs",
            'logging_steps':2,
            'save_steps':1000,
            'save_total_limit':1,
            'report_to':'none',
            'fp16':True,
            }
        

if __name__ == "__main__":  
    config=Config()