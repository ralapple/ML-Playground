'''
Created:        November 30th, 2023
Last Updated:   December 19th, 2023
Testing a simple image detection model, first time using pytorch
'''
# import numpy as np
import sys
import time

import torch

from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights



class ImageDetector:
    '''
    Image detection algorithm using pytorch
    Identify airplanes in an image
    Starting using a pretrained model from pytoch vision
    '''
    def __init__(self, model_path=None):
        '''
        Constructor for the ImageDetector class
        :param model_path: path to the model if there is one set, initialized to ResNet50 for default
        '''

        # if a model is provided, use that model
        if model_path:
            self.model = torch.load(model_path)

        else:
            # otherwise use the pretrained model from pytorch vision
            self.weights = ResNet50_Weights.IMAGENET1K_V2
            self.model = resnet50(weights=self.weights)

            # ROIHeads is the class for region of interest heads - handle the detection tasks
            # box_predictor is the class for the box predictor - predicts the bounding box coordinates around the object and class scores
            # cls_score is the classification layer of the box predictor - predicts the class scores for each region proposal
            # in_features is the variable that stores the number of input features to the classification layer
                # It's important because when you modify the model (in the line below), you need to know the number of input features for the new layer you are replacing it with.
            
            # sets the model to eval mode
            self.model.eval()

        self.batch = None

    def train(self) -> None:
        '''
        Training function for the model when creating our own CNN
        :return: None
        '''
        pass


    def predict(self, image_path, num_predictions=5) -> None:
        '''
        Predicts the bounding boxes for the image and casts them on the image
        :param image_path: path to the image
        :param num_predictions: number of predictions to make
        :return: None
        '''
        start_time = time.time()
        self.preprocess(image_path)

        prediction = self.model(self.batch).squeeze().softmax(0)

        # top k probabilities, top k indices within the prediction tensor
        topk_probability, topk_indices = torch.topk(prediction, num_predictions)

        # iterate and print the top k predictions 
        for i in range(num_predictions):
            label = topk_indices[i].item()
            score = topk_probability[i].item()
            category = self.weights.meta["categories"][label]
            print(f'Class: {category} | Probability: {score * 100:.2f}')

        print(f'Prediction time: {time.time() - start_time:.2f} seconds')


    def preprocess(self, image_path) -> None:
        '''
        Preprocesses the image before the model can make a prediction
        :param image_path: path to the image
        :return: None 
        '''

        # Open the image
        image = read_image(image_path)
        preprocess = self.weights.transforms(antialias=True)
        # preprocess the image
        batch = preprocess(image).unsqueeze(0)
        self.batch = batch



if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    num_predictions = 5 # default number of predictions

    if len(sys.argv) == 2:
        img = str(sys.argv[1])
    if len(sys.argv) == 3:
        img = str(sys.argv[1])
        num_predictions = int(sys.argv[2])

    Detector = ImageDetector()
    Detector.predict(img, num_predictions)
