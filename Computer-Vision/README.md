# Image Detection program

### Background
- Developed using the Pytorch Torchvision library
- Simple one image classification model
- Utilizes the Pytorch pretrained image model with pretrained weights

### Currently Working
- Reads image and determines top k possibilities for the class of the image

##### To Run
```
python ImageDetector.py {YOUR IMAGE PATH} {NUMBER OF PREDICTIONS: DEFAULT 5}
```

### Next Goals
- Expantion into highlighting the objects within the image
- Allowing for training of own weights
- decreasing coupling between components of the class

# Image Object Detector
- Reads in an image similar to the Image Detection
- Outlines the objects and their label within the image