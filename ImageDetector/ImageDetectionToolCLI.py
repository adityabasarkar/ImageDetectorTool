# Imports
##################################
import torch, cv2, os, random, json, time, sys
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models.vgg import VGG16_Weights
from torchsummary import summary
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler
from PIL import Image
import VideoCapture as vc
import albumentations as alb
from labelme import __main__ as labelme_main
##################################





# Variable Definitions
##################################
script_dir = os.path.dirname(os.path.abspath(__file__))
PicsDir = os.path.join(script_dir, 'Data', 'Training Set', 'Input')
LabelDir = os.path.join(script_dir, 'Data', 'Training Set', 'Labelled')
images = []
classification_targets = []
regression_targets = []
##################################





# Albumentation and Transformation
##################################
augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
                         alb.HorizontalFlip(p=0.5), 
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2), 
                         alb.RGBShift(p=0.2), 
                         alb.VerticalFlip(p=0.5)], 
                         bbox_params=alb.BboxParams(format='albumentations', 
                                                   label_fields=['class_labels'])
                        )

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
##################################





# Model Architure (VGG16 implementation)
##################################
class FaceDetector(nn.Module):
    def __init__(self):
        super(FaceDetector, self).__init__()

        self.vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)  # Assuming you meant pretrained weights
        self.vgg16 = nn.Sequential(*list(self.vgg16.children())[:-1])
        self.ReLU = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool2d((2, 2))

        self.classification = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid())
        
        self.regression = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4))        

    def forward(self, x):
        features = self.vgg16(x)
        features = self.global_max_pool(features)
        features = features.view(features.size(0), -1)
        features = self.ReLU(features)
        
        class_output = self.classification(features)
        reg_output = self.regression(features)

        return class_output, reg_output
##################################





# custom LR Decay scheduler
##################################
class InverseTimeDecayLR(_LRScheduler):
    def __init__(self, optimizer, decay_rate, last_epoch=-1):
        self.decay_rate = decay_rate
        super(InverseTimeDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr / (1.0 + self.decay_rate * self._step_count) for base_lr in self.base_lrs]
##################################





# custom Localization loss
##################################
class LocalizationLoss(nn.Module):
    def __init__(self):
        super(LocalizationLoss, self).__init__()

    def forward(self, yhat, y_true):
        delta_coord = torch.sum((y_true[:, :2] - yhat[:, :2])**2)
        
        h_true = y_true[:, 3] - y_true[:, 1]
        w_true = y_true[:, 2] - y_true[:, 0]
        h_pred = yhat[:, 3] - yhat[:, 1]
        w_pred = yhat[:, 2] - yhat[:, 0]
        
        delta_size = torch.sum((w_true - w_pred)**2 + (h_true - h_pred)**2)
        
        return delta_coord + delta_size
##################################





# Prepare model and CUDA device 
# If your device does not have a gpu or doesn't have CUDA, change the "cuda" parameter to "cpu"
##################################
model = FaceDetector()

device = torch.device("cpu")
if torch.cuda.is_available:
    torch.cuda.empty_cache()
    device = torch.device("cuda")

model = model.to(device)

dvc = next(model.parameters()).device
print("Model is on device:", dvc)
##################################





# Switch case
##################################
state = 1
while True:
    match state:
        case 1:
            kn = ""
            kn = input("\nTRAIN or LOAD model?\nOptions: [TRAIN, LOAD, exit]\n")
            if kn == "TRAIN":
                state = 2
            elif kn == "LOAD":
                state = 6
            elif kn == "exit":
                exit()
            else: 
                state = 1
        case 2:
            kn = ""
            print("\nThe input folder should only include files of type .jpg\nor .png pictures with 640 x 480 (SD) resolution")
            for x in os.listdir(PicsDir):
                if x.split(".")[1] != "jpg" and x.split(".")[1] != "png":
                    print("\nPlease modify the input folder to only include files of\ntype .jpg or .png pictures with 640 x 480 (SD) resolution")
                    time.sleep(1.5)
                    state = 1
            if (len(os.listdir(PicsDir)) < 30):
                print("\nThere needs to be atleast 30 images (.jpg or .png)\nof resolution (640 x 480) in the input directory.")
                kn = input("\nWould you like to take pictures using your webcam\nor move the images into the input folder manually?\nOptions: [WEBCAM, IMPORT, exit]\n")
                if kn == "WEBCAM":
                    vc.capturePics(os.path.join(script_dir, 'Data', 'Training Set', 'Input'))
                    state = 2
                elif kn == "IMPORT":
                    print("\nPlease place your images in the Data/Training Set/Input folder")
                    time.sleep(1.5)
                    state = 1
                elif kn == "exit":
                    exit()
                else: 
                    state = 2
            if (len(os.listdir(PicsDir)) > 30):
                print("\nYou have met the minimum data requirements")
                kn = input("\nAdd more images with webcam, import images, or next?\nOptions: [WEBCAM, IMPORT, NEXT, exit]\n")
                if kn == "WEBCAM":
                    print("\nPress Q to exit camera\n")
                    time.sleep(2.5)
                    print("\nIt might take a few seconds for the camera to open\n")
                    vc.capturePics(os.path.join(script_dir, 'Data', 'Training Set', 'Input'))
                    state = 2
                elif kn == "IMPORT":
                    print("\nPlease place your images in the Data/Training Set/Input folder\n")
                    time.sleep(1.5)
                    state = 1
                elif kn == "NEXT":
                    state = 3
                elif kn == "exit":
                    exit()
                else:
                    state = 2
        
        case 3:
            kn = ""
            kn = input("\nDo you have labelme installed on your machine?\nOptions: [YES, NO, exit]\n")
            if kn == "YES":
                state = 4
            elif kn == "NO":
                print("\nPlease open the command prompt and install labelme using\nthe following prompt: pip install labelme")
                state = 1
            elif kn == "exit":
                exit()
            else:
                state = 3
        
        case 4:
            kn = ""
            kn = input("\nHave you annotated your images using labelme? The only\nannotation this model accepts are rectangles.\nYou can create these annotations by entering ctrl+R on your\nkeyboard, or by going to Edit -> Create Rectangle\nOptions: [YES, NO, exit]\n")
            if kn == "YES":
                state = 5
            elif kn == "NO":
                print("\n - Set input directory to the input folder.\n - Set the output directory to the labelled folder.\n - Go to File -> Save Automatically.\n - Use Edit -> Create Rectangle to create annotations.\n - Close the program when finished. All json files should be in the labelled folder.\n - Restart the Image Detection Tool program to continue.\n")
                time.sleep(1)
                print("\nOpening labelme\n")
                labelme_main.main()
                state = 4
            elif kn == "exit":
                exit()
            else: 
                state = 4

        case 5:
            print("\nUploading data to data collector lists...\n")
            time.sleep(1.5)

            for file in os.listdir(PicsDir):
        
                jsonFileName = file.split(".")[0]
                jsonFilePath = os.path.join(LabelDir, jsonFileName + ".json")

                if os.path.exists(os.path.join(LabelDir, jsonFileName + ".json")):
                    with open(os.path.join(LabelDir, jsonFileName + ".json"), 'r') as f:
                        data = json.load(f)

                    x1 = data['shapes'][0]['points'][0][0]/640
                    y1 = data['shapes'][0]['points'][0][1]/480
                    x2 = data['shapes'][0]['points'][1][0]/640
                    y2 = data['shapes'][0]['points'][1][1]/480

                    if x2 < x1:
                        new = x1
                        x1 = x2
                        x2 = new

                    if y2 < y1:
                        new = y1
                        y1 = y2
                        y2 = new

                    coordinateList = [x1, y1, x2, y2] # input into albumentations
                    
                    # Read Image
                    image = cv2.imread(os.path.join(PicsDir, file))

                    for i in range(60):
                        # Convert Image to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Get the augmented data
                        augmented_data = augmentor(image=image, bboxes=[coordinateList], class_labels=['face'])
                        
                        # Add augmented image to images list
                        augmented_image = augmented_data['image']
                        augmented_image_pil = Image.fromarray(np.uint8(augmented_image)).convert('RGB')
                        transformed_image = transform(augmented_image_pil)
                        images.append(transformed_image)

                        # Add augmented coordinates to coordinate list
                        if len(augmented_data['bboxes']) == 0:
                            regression_targets.append(torch.tensor([0, 0, 0, 0]))
                            classification_targets.append(torch.tensor([0]))
                            #augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                            #cv2.imwrite(os.path.join(script_dir, "Data", "Training Set", "AugmentedImageSamples", "augIm_" + str(random.randint(11111,99999))) + ".jpg", augmented_image_bgr)
                        else: 
                            aug_coords = list(augmented_data['bboxes'][0])
                            regression_targets.append(torch.tensor(aug_coords))
                            classification_targets.append(torch.tensor([1]))
                            #augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                            #cv2.imwrite(os.path.join(script_dir, "Data", "Training Set", "AugmentedImageSamples", "augIm_" + str(random.randint(11111,99999))) + ".jpg", augmented_image_bgr)
                        
                        # Add classification to classification list
                        #classification_targets.append(torch.tensor([1]))

                else:
                    
                    x1 = 0
                    y1 = 0
                    x2 = 0.00001
                    y2 = 0.00001

                    coordinateList = [x1, y1, x2, y2]

                    # Read Image
                    image = cv2.imread(os.path.join(PicsDir, file))

                    for i in range(60):    
                        # Convert image to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Get the augmented data
                        augmented_data = augmentor(image=image, bboxes=[coordinateList], class_labels=['face'])
                        
                        # Add augmented image to images list
                        augmented_image = augmented_data['image']
                        augmented_image_pil = Image.fromarray(np.uint8(augmented_image)).convert('RGB')
                        transformed_image = transform(augmented_image_pil)
                        images.append(transformed_image)
                        
                        classification_targets.append(torch.tensor([0]))
                        regression_targets.append(torch.tensor(coordinateList))

            print("\nUploading data lists to dataloader for training...\n")
            # Dataloader
            class ImageDataset(Dataset):
                def __init__(self, image_list, classification_targets, regression_targets):
                    self.image_list = image_list
                    self.classification_targets = classification_targets
                    self.regression_targets = regression_targets

                def __len__(self):
                    return len(self.image_list)

                def __getitem__(self, index):
                    # Get the image
                    image = self.image_list[index]
                    classification_target = self.classification_targets[index]
                    regression_target = self.regression_targets[index]

                    return image, (classification_target, regression_target)

            dataset = ImageDataset(images, classification_targets, regression_targets)
            data_loader = DataLoader(dataset, batch_size=5, shuffle=True)

            model.train()

            criterion_classification = nn.BCELoss()
            criterion_regression = LocalizationLoss()

            learningRate = 1e-4
            BPE = len(data_loader)
            decayRate = (1/0.75 - 1)/BPE
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scheduler = InverseTimeDecayLR(optimizer, decay_rate=decayRate)

            kn = ""
            num_epochs = ""
            while not(isinstance(num_epochs, int)):
                try: 
                    kn = input("\nHow many epochs would you like to train? (Enter a number)\n")
                    if kn == "exit":
                        exit()
                    num_epochs = int(kn)
                except Exception as e:
                    print("\nPlease enter a number\n")
            
            kn = ""
            kn = input("\nContinue training existing model?\nOptions: [YES, NO, exit]\n")
            if kn == "YES":
                
                kn = ""
                saved_model_path = os.path.join(script_dir, "Custom Models")

                accumulate = 0
                for x in os.listdir(saved_model_path):
                    if x.split(".")[1] == "pt":
                        accumulate += 1

                if accumulate == 0:
                    print("\nThere are no saved models in the Custom Models directory\n")

                elif accumulate > 0:
                    print("\nHere are a list of all models in your directory:")
                    for x in os.listdir(saved_model_path):
                        if x.split(".")[1] == "pt":
                            print(f" - {x}")
                    
                    k = 0
                    while k == 0:
                        try:
                            kn = ""
                            kn = input("\nWhich model do you want to upload?\nEnter just the name of the file, not the file extension\n")
                            if kn == "exit":
                                exit()
                            while not((kn + ".pt") in os.listdir(saved_model_path)):
                                print("\nModel either doesn't exist in folder or is not supported by this program\n")
                                kn = input("\nWhich model do you want to upload?\n")
                                if kn == "exit":
                                    exit()
                            
                            modelName = kn + ".pt"
                            model = FaceDetector()
                            state_dict = torch.load(os.path.join(saved_model_path, modelName))
                            model.load_state_dict(state_dict)
                            k = 1

                        except Exception as e:
                            print("\nModel either doesn't exist in folder or is not supported by this program\n")
                            k = 0

                    if torch.cuda.is_available:
                        device = torch.device("cuda")
                    else:
                        device = torch.device("cpu")
                    
                    model = model.to(device)
                    print("\nModel has been uploaded, starting training...\n")
            
            elif kn == "NO":
                print("\nStarting training...\n")
            
            elif kn == "exit":
                exit()
            
            alpha = 0
            beta = 0

            kn = ""
            while kn == "":
                try:
                    kn = input("\nEnter classification loss hyperparameter\nOptions: [a number, DEF, exit] (DEF - default: 1)\n")
                    if kn == "DEF":
                        alpha = 1
                    elif kn == "exit":
                        exit()
                    else:
                        alpha = int(kn)
                except Exception as e:
                    print("\nEnter a valid input\n")
            
            kn = ""
            while kn == "":
                try:
                    kn = input("\nEnter regression loss hyperparameter\nOptions: [a number, DEF, exit] (DEF -> default: 1)\n")
                    if kn == "DEF":
                        beta = 1
                    elif kn == "exit":
                        exit()
                    else:
                        beta = int(kn)
                except Exception as e:
                    print("\nEnter a valid input\n")
            
            for epoch in range(num_epochs):
        
                for i, (images, (classification_targets, regression_targets)) in enumerate(data_loader):
                    
                    images = images.to(device)
                    classification_targets = classification_targets.to(device).float()
                    regression_targets = regression_targets.to(device)
                    
                    optimizer.zero_grad()

                    # Forward pass
                    class_output, reg_output = model(images)

                    # Loss for classification and regression
                    loss_classification = criterion_classification(class_output.squeeze(), classification_targets.squeeze())
                    loss_regression = criterion_regression(reg_output, regression_targets)
                    
                    
                    # Combine losses and backpropagate
                    total_loss = alpha * loss_classification + beta * loss_regression
                    total_loss.backward()

                    # Update the model parameters
                    optimizer.step()
                    scheduler.step()
                    
                    print(f"Epoch [{epoch+1}/{num_epochs}], Progress: [{i+1}/{len(data_loader)}], Class_Loss: {round(loss_classification.item(), 5)}, Reg_Loss: {round(loss_regression.item(), 5)}, Loss: {round(total_loss.item(), 5)}")
                    

            save_folder = os.path.join(script_dir, "Custom Models")
            kn = ""
            k = 0
            while k == 0:    
                kn = input("\nWhat would you like to name your model?\n")
                if kn == "exit":
                    exit()
                
                try:
                    model_filename = kn + ".pt"
                    torch.save(model.state_dict(), os.path.join(save_folder, model_filename))
                    print(f"Model has been trained and saved to the 'Custom Models' folder as {model_filename}")
                    k = 1
                except Exception as e:
                    print("\nInvalid file name\n")
                    k = 0
            
            state = 6
        

        case 6:
            kn = ""
            saved_model_path = os.path.join(script_dir, "Custom Models")

            accumulate = 0
            for x in os.listdir(saved_model_path):
                if x.split(".")[1] == "pt":
                    accumulate += 1

            if accumulate == 0:
                print("\nThere are no saved models in the Custom Models directory\n")

            elif accumulate > 0:
                print("\nHere are a list of all models in your directory:")
                for x in os.listdir(saved_model_path):
                    if x.split(".")[1] == "pt":
                        print(f" - {x}")
            
                
                
                k = 0
                while k == 0:
                    try:
                        kn = ""
                        kn = input("\nWhich model do you want to use?\nEnter just the name of the file, not the file extension\n")
                        if kn == "exit":
                            exit()
                        while not((kn + ".pt") in os.listdir(saved_model_path)):
                            print("\nModel either doesn't exist in folder or is not supported by this program\n")
                            kn = input("\nWhich model do you want to upload?\nEnter just the name of the file, not the file extension\n")
                            if kn == "exit":
                                exit()
                        
                        modelName = kn + ".pt"
                        model = FaceDetector()
                        state_dict = torch.load(os.path.join(saved_model_path, modelName))
                        model.load_state_dict(state_dict)
                        k = 1

                    except Exception as e:
                        print("\nModel either doesn't exist in folder or is not supported by this program\n")
                        k = 0

                if torch.cuda.is_available:
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")
                
                model = model.to(device)

                print("\nModel has been uploaded from folder.\n")

                kn = ""
                cap = cv2.VideoCapture(0)
                kn = input("\nWould you like to open camera?\nOptions: [YES, NO, exit]\n")
                if kn == "YES":
                    print("\nPress Q to Exit Camera\n")

                    time.sleep(2.5)

                    if not cap.isOpened():
                        print("\nError opening camera.\n")

                    while True:
                        ret, frame = cap.read()  # Capture frame-by-frame

                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
                        transformed_image = transform(image_pil)

                        # Add batch dimension to the transformed image
                        transformed_image = transformed_image.unsqueeze(0)

                        # Move the input tensor to the GPU
                        transformed_image = transformed_image.to(device)
                        model.eval()

                        with torch.no_grad():
                            class_output, reg_output = model(transformed_image)

                        height, width, _ = frame.shape
                        start_point = (int(reg_output[0][0].item() * width), int(reg_output[0][1].item() * height))
                        end_point = (int(reg_output[0][2].item() * width), int(reg_output[0][3].item() * height))
                        color = (255, 0, 0)
                        thickness = 2

                        if not ret:
                            print("Error capturing frame.")
                            break
                        if (class_output[0].item() > 0.5):
                            frame_with_box = cv2.rectangle(frame.copy(), start_point, end_point, color, thickness)
                            cv2.imshow('Camera with Bounding Box', frame_with_box)
                        else:
                            cv2.imshow('Camera with Bounding Box', frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == ord('Q'):  # Press 'q' to quit the camera window
                            break
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    state = 1

                elif kn == "NO":
                    state = 1
                    cap.release()
                    cv2.destroyAllWindows()
                
                elif kn == "exit":
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            
            







##################################