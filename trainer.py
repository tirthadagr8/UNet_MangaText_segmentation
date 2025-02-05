from config import Config
from transformers import Trainer, TrainingArguments

def get_trainer(config:Config, model, train_dataset, neptune_callback=None)->Trainer:
    """
    Returns a Trainer object for training a model on a dataset.
    
    Args:
        model (torch.nn.Module): Model to train.
        train_dataset (torch.utils.data.Dataset): Training dataset.
        neptune_callback (Optional[NeptuneCallback]): Neptune callback for logging metrics.
    
    Returns:
        Trainer: Trainer object for training the model.
    """
    if config.report_to=='neptune':
        from transformers.integrations import NeptuneCallback
        neptune_callback = [NeptuneCallback(
            project=config.NEPTUNE_PROJECT,
            api_token=config.NEPTUNE_API_TOKEN
        )]
    training_args=TrainingArguments(**config.trainer_config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=None,  # Default collator works for VisionEncoderDecoderModel
        callbacks=neptune_callback
    )
    return trainer