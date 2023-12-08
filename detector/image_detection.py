'''
Created:        November 30th, 2023
Last Updated:   December 7th, 2023
Testing a simple image detection model, first time using pytorch
'''
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes


"""
Image detection algorithm using pytorch
"""
class AirplaneDetector:
    def __init__(self, model_path=None, num_classes=91):

        # if a model is provided, use that model
        if model_path:
            self.model = torch.load(model_path)
        else:
            # otherwise use the pretrained model from pytorch vision
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)

            # ROIHeads is the class for region of interest heads - handle the detection tasks
            # box_predictor is the class for the box predictor - predicts the bounding box coordinates around the object and class scores
            # cls_score is the classification layer of the box predictor - predicts the class scores for each region proposal
            # in_features is the variable that stores the number of input features to the classification layer
                # It's important because when you modify the model (in the line below), you need to know the number of input features for the new layer you are replacing it with.
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features

            # 
            self.model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, num_classes + 1)
        
        self.model.eval()
        self.transforms = T.Compose([T.ToTensor()])

    def train(self):
        '''
        Training function for the model when creating our own CNN
        '''
        pass

    def predict(self, image_path):
        '''
        Predicts the bounding boxes for the image
        '''

        # Preprocess the image using function below
        image_array = preprocess_image(image_path)

        # Convert the image to a tensor
        image_tensor = self.transforms(image_array)

        # Change the tensor values to floats - for the pytorch model
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

    # Normalize the image values
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