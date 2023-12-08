'''
Created:        November 30th, 2023
Last Updated:   November 30th, 2023
Testing a simple image detection model, first time using pytorch
'''
import os
from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes




class AirplaneDetector:
    def __init__(self, model_path=None, num_classes=91):
        if model_path:
            self.model = torch.load(model_path)
        else:
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, num_classes + 1)
        
        self.model.eval()
        self.transforms = T.Compose([T.ToTensor()])


    def train(self):
        pass

    def predict(self, image_path):
        image_array = preprocess_image(image_path)
        image_tensor = self.transforms(image_array)
        image_tensor = image_tensor.float()
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Pass the image to the model
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Print model predictions
        print(predictions)

        # Extract bounding boxes and labels
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        # Visualize bounding boxes on the image
        image_with_boxes = draw_bounding_boxes(image_array, boxes, labels)
        Image.fromarray(image_with_boxes.mul(255).permute(1, 2, 0).byte().numpy()).show()

    def evaluate(self):

        pass




def preprocess_image(image_path):
    new_min = 0
    new_max = 255

    # Open image and create an array of the pixel values
    image = Image.open(image_path)
    
    # Convert the image to RGB (remove alpha channel if present)
    image = image.convert("RGB")
    
    image_arr = np.array(image)

    # Normalize the values of the image
    old_min, old_max = image_arr.min(), image_arr.max()
    image_arr = (image_arr - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min

    return image_arr


def convert_image(image_arr):
    '''
    converts the image to useable format for the model
    param: image_arr: array of the image
    '''

    # Convert the image to a tensor
    pass


if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    AirplaneDetector(model_path=None).predict('./airplane.png')