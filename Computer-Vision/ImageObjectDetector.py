'''
ImageObjectDetector.py
Detects objects within an object and outlines
'''

from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


class ImageObjectDetector:
    '''
    Image object detection algorithm using pytorch
    '''
    def __init__(self, input_model = None) -> None:
        '''
        Constructor for the ImageObjectDetector class
        '''

        if input_model:
            
            # Set the model to a known path
            self.model = torchvision.load(input_model)
        else:

            # Pretrained Pytorch weights
            self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT

            # Initialize a Pytorch model with the pretrained weights and threshold of the object
            self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=0.8)

        # Initialize the batch
        self.batch = None


    def predict(self, image_path) -> None:
        '''
        Predicts the bounding boxes for the image and casts them on the image
        :param image_path: path to the image
        :param num_predictions: number of predictions to make
        :return: None
        '''

        # Set the model to eval mode, preprocess the image
        self.model.eval()
        self.preprocess(image_path)

        # Make the prediction
        prediction = self.model(self.batch)[0]

        # Retrieve the labels for the objects within the prediction
        labels = [self.weights.meta["categories"][i] for i in prediction["labels"]]

        # Edit the tensor to have the boxes and labels around the objects
        box = draw_bounding_boxes(read_image(image_path), prediction["boxes"], labels=labels, colors="red", width=2)

        # Convert the tensor back to an image and display
        new_image = to_pil_image(box)
        new_image.show()


    def preprocess(self, image_path):
        '''
        Uses the weight's transforms to preprocess the image for the model
        :param image_path: path to the image
        '''

        # Opens the image
        img = read_image(image_path)

        # Transforms the image for the model
        preprocess_function = self.weights.transforms()

        # Sets the batch to the preprocessed image
        self.batch = [preprocess_function(img)]


if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    detector = ImageObjectDetector()
    detector.predict("./images/bird_with_airplane.jpg")