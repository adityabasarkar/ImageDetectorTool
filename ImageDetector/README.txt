Image Detection Tool CLI program

This is a simple image detection tool program that can be used to detect any object 
(including faces) and puts a bounding box around it. It uses pytorch and leverages
the pretrained VGG16 model. This is a limited tool as it takes in very specific input
types (.jpg or .png files, 640 x 480p images) and can only show one bounding box at a time. 

This program can be used to train new models or load/train existing models that use
the same architecture as this program. 

To use the program, open ImageDetectionToolCLI.py

How to use the program:

Input Images
-------------------------------
All input images have to be of SD resolution (640 x 480) and have
to be placed in the Data/Training Set/Input folder. Images can be
placed from external sources or the program can open up the webcam
in SD resolution to take photos. If you choose to open up the webcam,
press s to take pictures and q to quit and continue the program. The 
more pictures taken, the better. 

Labelling
-------------------------------
Input images have to be labelled through the labelme software. The 
labels have to be placed in the Data/Training Set/Labelled folder.
The folder can be left blank, although training the model will most
likely have no effect. If you choose to label the input images through
labelme, follow these steps:

    1. install labelme (pip install labelme)
    2. open labelme through the command prompt by entering "labelme"
    3. Open the folder containing the input images (Open dir -> "Input" folder)
    4. Change the output folder to the "Labelled" folder
    5. Press File -> Save Automatically to automatically save all labels
    6. To start labelling the images, go to edit -> create rectangle.
    7. Draw the rectangle around the object that you want to detect (one label per image).
    8. Press A or D to either go to the previous or next image
    9. close the labelme software once finished

All labels should be located in the "Labelled" folder as json files. 

Training/Loading Models
-------------------------------
The CLI does give you the option to change the number of epochs. However, the
batch size, optimizer, and loss function have to be changed through the codebase. 
Once the program finishes training the model, it saves it to the "Custom Models" 
folder as "ImageDetectorModel.pt". If the model is being loaded, it uses the existing 
ImageDetectorModel.pt model from the "Custome Models" folder

If you do not have a GPU and the CUDA computing toolkit on your machine, the program
trains the model using the CPU. However, having a GPU and CUDA computing toolkit helps 
to train the model significantly faster. Make sure the CUDA computing toolkit and the 
PyTorch version installed on your computer are compatible.

Default Training parameters
-------------------------------
Batch Size: 5
Regression Loss function: Custom Localization Loss
Regression Loss Hyperparameter: 1.9
Classification Loss function: BCELoss (Binary Cross Entropy)
Classification Loss Hyperparameter: 1.1
Optimizer: Adam
Scheduler: Inverse Time Decay (Custom)
Epochs: Decided by user
Learning Rate: 1e-4
LR_Decay: (1/0.75 - 1)/(batches per epoch)

Note: the hyperparameters for training can vary based on the complexity of the problem
as well as other factors. You may need to find ways to optimize hyperparameters. 

Architecture
-------------------------------

                                           /----> Classification Model (6 FC Layers) (Outputs 0 or 1)
Inputs (640 x 480 RGB image) --> VGG16 -->|
                                           \----> Regression Model (6 FC Layers) (Outputs x1, y1, x2, y2)

Using the Model
-------------------------------
Once the model has been trained or loaded the program gives an option to open the camera.
Select the model you want to use and once given the option by the CLI, open the camera. There
should be a blue bounding box that tracks the object the model was trained to track. 

IPT4
-------------------------------
Batch Size: 5
Regression Loss function: Custom Localization Loss
Classification Loss function: BCELoss (Binary Cross Entropy)
Optimizer: Adam
Scheduler: Inverse Time Decay (Custom)
Epochs: 2
Learning Rate: 1e-4
LR_Decay: (1/0.75 - 1)/(batches per epoch)
