import cv2
import torch
import numpy as np
from PIL import Image
from config import Config
from typing import Union
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.patches as patches
from torchvision import transforms as T
from transformers.utils import ModelOutput

config=Config()

@dataclass
class SegmentationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


# Helper function to count trainable parameters
def count_parameters(model): # print(f"Trainable parameters: {count_parameters(model)/1e6:.1f}M")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images)  # Batch images
    return images, targets

def combined_dice_bce_loss(predicted_mask, input_mask, size=512, bce_weight=1, dice_weight=0.0):
    """
    Computes a combined loss of Dice Loss and BCEWithLogitsLoss.
    
    Args:
        predicted_mask (torch.Tensor): Predicted mask logits, shape (batch_size, num_boxes, height_pred, width_pred).
        input_mask (torch.Tensor): Ground truth binary mask, shape (batch_size, num_boxes, height_gt, width_gt).
        size (int): Target size for resizing the input mask to match the predicted mask's resolution.
        bce_weight (float): Weight for BCEWithLogitsLoss in the combined loss.
        dice_weight (float): Weight for Dice Loss in the combined loss.
    
    Returns:
        torch.Tensor: Combined loss value.
    """
    # Step 1: Resize the input mask to match the predicted mask's resolution
    input_mask_resized = F.interpolate(
        input_mask.float(), 
        size=(size, size),          # Match the predicted mask's resolution
        mode='nearest'              # Use 'nearest' for binary masks
    )

    # Step 2: Flatten both masks
    input_mask_flat = input_mask_resized.view(-1)       # Shape: (batch_size * num_boxes * height_pred * width_pred)
    predicted_mask_flat = predicted_mask.view(-1)       # Shape: (batch_size * num_boxes * height_pred * width_pred)

    # Step 3: Compute BCEWithLogitsLoss
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss instead of BCELoss
    bce_loss = bce_loss_fn(predicted_mask_flat, input_mask_flat)

    # Step 4: Compute Dice Loss
    def dice_loss(pred, target, smooth=1.0):
        pred_sigmoid = torch.sigmoid(pred)  # Convert logits to probabilities
        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.sum() + target.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice  # Return the complement of the Dice coefficient

    dice_loss_value = dice_loss(predicted_mask_flat, input_mask_flat)

    # Step 5: Combine the losses
    combined_loss = bce_weight * bce_loss + dice_weight * dice_loss_value

    return combined_loss


def extract_bounding_boxes_from_masks(masks, threshold=0.5):
    """
    Extracts bounding boxes from predicted masks.
    
    Args:
        masks (torch.Tensor): Predicted masks with shape (batch_size, num_classes, height, width).
                              Values are probabilities (0 to 1).
        threshold (float): Threshold to binarize the mask. Pixels with values >= threshold are considered part of the object.
    
    Returns:
        List[List[Dict]]: A list of bounding boxes for each batch and class.
                          Each bounding box is represented as a dictionary with keys:
                          - "x_min": Top-left x-coordinate
                          - "y_min": Top-left y-coordinate
                          - "x_max": Bottom-right x-coordinate
                          - "y_max": Bottom-right y-coordinate
                          - "class_id": Class ID corresponding to the mask
    """
    # Ensure masks are on CPU and converted to numpy for processing
    masks = masks.detach().cpu().numpy()
    
    batch_bboxes = []  # To store bounding boxes for all batches
    
    for batch_idx in range(masks.shape[0]):  # Iterate over each batch
        bboxes_for_batch = []
        
        for class_id in range(masks.shape[1]):  # Iterate over each class
            mask = masks[batch_idx, class_id]  # Get the mask for the current class
            
            # Binarize the mask using the threshold
            binary_mask = (mask >= threshold).astype(np.uint8)
            
            if binary_mask.sum() == 0:  # Skip if no object is detected in the mask
                continue
            
            # Find contours (connected components) in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get the bounding box coordinates
                x_min, y_min, w, h = cv2.boundingRect(contour)
                x_max = x_min + w
                y_max = y_min + h
                
                # Append the bounding box information
                bboxes_for_batch.append({
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "class_id": class_id
                })
        
        batch_bboxes.append(bboxes_for_batch)
    
    return batch_bboxes

def plot_box(img:Union[torch.Tensor,Image.Image],box:list):
    if isinstance(img,torch.Tensor):
        if len(img.shape)==4:
            img=T.Resize(config.img_size)(img[0])
        img=img.permute(1,2,0).detach().cpu().numpy()
    elif isinstance(img,Image.Image):
        img=np.array(img.convert('RGB').resize((config.img_size,config.img_size)))
    fig,ax=plt.subplots()
    plt.imshow(img)
    for b in box:
        rect=patches.Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    plt.show()

def inference(model,image_path,img_size=512,device='cuda'):
    img=Image.open(image_path).convert('RGB')
    simple_transforms=T.Compose([
            T.Resize((config.img_size,config.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    img=simple_transforms(img).unsqueeze(0).to(device)
    model.eval().to(device)
    with torch.no_grad():
        outputs = model(img).logits
    plt.imshow(img[0].permute(1,2,0).detach().cpu())
    plt.show()
    plt.imshow(outputs[0].permute(1,2,0).detach().cpu(),cmap='gray')
    plt.show()
    boxes=extract_bounding_boxes_from_masks(outputs.detach().cpu())
    plot_box(img,boxes[0])
    return outputs
