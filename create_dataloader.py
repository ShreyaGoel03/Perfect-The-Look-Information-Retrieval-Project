import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torchvision
import random
from PIL import Image
import pickle
from google.colab.patches import cv2_imshow
from urllib.request import urlopen
import urllib.error

class fashionDataset(Dataset):
  def __init__(self,csv_file,transform=None):
    self.meta_data = pd.read_csv(csv_file)    
    self.transform = transform
  
  def __len__(self):
    return len(self.meta_data)

  def __getitem__(self,index):
    scene_url = self.meta_data.iloc[index,7]
    product_url = self.meta_data.iloc[index,8]    
    scene_img = io.imread(scene_url)
    scene_img = cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB)
    p_img = io.imread(product_url)       
    p_img = cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB) 
    #generating negative product image
    category = self.meta_data.iloc[index,6]    
    category_indices = self.meta_data.index[self.meta_data['category']==category].tolist()    
    product_id = self.meta_data.iloc[index,0]        
    neg_list = [ind for ind in category_indices if ind != index and self.meta_data.iloc[ind,0] != product_id]    
    if len(neg_list) == 0:
      print("empty_list")
    neg_img_index = random.choice(neg_list)
    neg_img_url = self.meta_data.iloc[neg_img_index,8]
    neg_img = io.imread(neg_img_url) 
    neg_img = cv2.cvtColor(neg_img, cv2.COLOR_BGR2RGB)  
    if self.transform is not None:
         arg = [scene_img,index]
         p_img,neg_img = self.transform[0](p_img),self.transform[0](neg_img)            
         scene_img = self.transform[1](arg)                 
    return scene_img,p_img,neg_img,category

#cropping products from scene image based on bounding box coordinates
class CustomCrop(object):
  def __init__(self,csv_file):
    self.meta_data = pd.read_csv(csv_file)
  def __call__(self,sample):    
    scene,index = sample[0],sample[1]
    h, w = scene.shape[:2]
    top,bottom = self.meta_data.iloc[index,3]*h-0.05*h,self.meta_data.iloc[index,5]*h+0.05*h    
    cropped_scene_top = scene[:int(top),:] 
    cropped_scene_bottom = scene[int(bottom):,:] 
    area1 = cropped_scene_top.shape[0]*cropped_scene_top.shape[1]  
    area2 = cropped_scene_bottom.shape[0]*cropped_scene_bottom.shape[1]      
    if area1 > area2:
      return cropped_scene_top    
    return cropped_scene_bottom    

#Transformations on scene and product images
def transform_obj(path):
  #Product
  transf1 = transforms.Compose([          
      transforms.ToPILImage(),
      transforms.Resize((256,256)),  
      transforms.RandomCrop(224),
      transforms.RandomHorizontalFlip(),  
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  #Scene
  transf2 = transforms.Compose([      
      CustomCrop(path),  
      transforms.ToPILImage(),  
      transforms.Resize((256,256)),
      transforms.RandomCrop(224),
      transforms.RandomHorizontalFlip(),     
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  return transf1, transf2

class DataLoaders:
    def __init__(self):    
        train_path = 'data/STL-Dataset/train_data.csv'
        validate_path = 'data/STL-Dataset/validate_data.csv'
        test_path = 'data/STL-Dataset/test_data.csv'

        transf1_train, transf2_train = transform_obj(train_path)
        dataset_train = fashionDataset(train_path,transform=[transf1_train,transf2_train])

        transf1_val, transf2_val = transform_obj(validate_path)
        dataset_validate = fashionDataset(validate_path,transform=[transf1_val,transf2_val])

        transf1_test, transf2_test = transform_obj(test_path)
        dataset_test = fashionDataset(test_path,transform=[transf1_test,transf2_test]) 

        #Generating DataLoaders to feed into the model
        dataloader_train = DataLoader(dataset_train,batch_size=16)
        dataloader_validate = DataLoader(dataset_validate,batch_size=1)
        dataloader_test = DataLoader(dataset_test,batch_size=1)
        
        #Creating dictionary of dataloaders
        dataloaders = {}
        dataloaders['train'] = dataloader_train
        dataloaders['val'] = dataloader_validate
        dataloaders['test'] = dataloader_test
        self.dataloaders = dataloaders        
        dataset_sizes = {}
        dataset_sizes['train'] = len(dataset_train)
        dataset_sizes['val'] = len(dataset_validate)
        dataset_sizes['test'] = len(dataset_test)
        self.dataset_sizes = dataset_sizes
        
        
