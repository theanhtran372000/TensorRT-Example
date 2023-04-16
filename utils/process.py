import cv2
from albumentations import Resize, Compose
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import Normalize
import torch

def preprocess_image(img_path):
    # Transform for image
    transforms = Compose([
        Resize(224, 224, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Read image with cv2
    input_img = cv2.imread(img_path)
    
    # Transform image
    input_data = transforms(image=input_img)['image']
    
    # Convert to batch 1 image
    batch_data = torch.unsqueeze(input_data, 0)
    return batch_data


def postprocess_result(output_data, class_path):
    # Get class name
    with open(class_path) as f:
        classes = [line.split(',')[1].strip() for line in f.readlines()]
    
    # Calculate score  
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    
    # Find top predicted classes
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    
    while confidences[indices[0][i]] > 0.5:
        class_idx = indices[0][i]
        
        print(
            "Class:",
            classes[class_idx],
            ", confidence:",
            confidences[class_idx].item(),
            "%, index:",
            class_idx.item()
        )
        
        i += 1