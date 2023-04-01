#import libraries


import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from torch.autograd import Variable
import time
import sys
from torch import nn
from torchvision import models
import os
import uuid

#Model with feature visualization
from torch import nn
from torchvision import models
class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained = True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))




im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    cv2.imwrite('./2.png',image*255)
    return image

def predict(model,img,path = './'):
  fmap,logits = model(img.to('cpu'))
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _,prediction = torch.max(logits,1)
  confidence = logits[:,int(prediction.item())].item()*100
  #print('confidence of prediction:',logits[:,int(prediction.item())].item()*100)
  idx = np.argmax(logits.detach().cpu().numpy())
  bz, nc, h, w = fmap.shape
  out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  predict = out.reshape(h,w)
  predict = predict - np.min(predict)
  predict_img = predict / np.max(predict)
  predict_img = np.uint8(255*predict_img)
  out = cv2.resize(predict_img, (im_size,im_size))
  heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
  img = im_convert(img[:,-1,:,:,:])
  result = heatmap * 0.5 + img*0.8*255
  #cv2.imwrite('/content/1.png',result)
  # result1 = heatmap * 0.5/255 + img*0.8
  # r,g,b = cv2.split(result1)
  # result1 = cv2.merge((r,g,b))
  # plt.imshow(result1)
  # plt.show()
  return [int(prediction.item()),confidence]

def frame_extract(video):
    #print('Inside Extraction')
    vidObj = cv2.VideoCapture(video)
    success, image = vidObj.read()
    count = 0
    success = True
    while success:
        #print("Successs",success)
        success,image = vidObj.read()
        #print(image)
        count+=1
        if success:
            yield image

def validation_dataset(video, sequence_length = 20,transform = None):
    
    frames = []
    a = int(100/sequence_length)
    first_frame = np.random.randint(0,a)      
    for i,frame in enumerate(frame_extract(video)):
        #if(i % a == first_frame):
        faces = face_recognition.face_locations(frame)
        try:
          top,right,bottom,left = faces[0]
          frame = frame[top:bottom,left:right,:]
        except:
          pass
        frames.append(transform(frame))
        if(len(frames) == sequence_length):
          break
    frames = torch.stack(frames)
    frames = frames[:sequence_length]
    return frames.unsqueeze(0)

"""
def save_mp4_file(mp4_file):
    # Get the directory path of the code file
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Create the path to the uploaded_video directory
    video_dir_path = os.path.join(base_path, 'uploaded_video')

    # Create the uploaded_video directory if it doesn't exist
    if not os.path.exists(video_dir_path):
        os.makedirs(video_dir_path)

    # Generate a unique file name for the MP4 file
    file_name = str(uuid.uuid4()) + '.mp4'

    # Join the directory path and file name to create the full file path
    file_path = os.path.join(video_dir_path, file_name)

    # Save the MP4 file to the uploaded_video directory
    with open(file_path, 'wb') as f:
        f.write(mp4_file)

    # Return the full file path of the saved MP4 file
    return file_path"""


def model_prediction(file_path):
    
    im_size = 112
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    sm = nn.Softmax()
    inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
    train_transforms = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((im_size,im_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,std)])
    
    video = file_path
    video_dataset = validation_dataset(video,sequence_length = 20,transform = train_transforms)
    model = Model(2)
    path_to_model = "models.pt"
    model.load_state_dict(torch.load(path_to_model,map_location=torch.device('cpu')))
    model.eval()
    
    prediction = predict(model,video_dataset,'./')
    if prediction[0] == 1:
      #print("REAL")
      result = "REAL"
      return result
    else:
      #print("FAKE")
      result = "Fake"
      return result



#answer = model_prediction()
#print(answer)
