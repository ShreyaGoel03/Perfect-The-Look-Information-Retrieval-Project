import torchvision.models as models
import torch.nn as nn
import torch
from torchsummary import summary
from torchinfo import summary as summary1
from torch import optim
import time
import copy
from sklearn.metrics import roc_auc_score, accuracy_score,roc_curve
from tqdm.auto import tqdm
from statistics import mean
import matplotlib.pyplot as plt
from category_embedding.py import *

class Train:
    def __init__(self):
        pass
    def compute_compatability(x1,x2):  
        pdist = nn.PairwiseDistance(p=2)
        d_global = pdist(x1[0],x2)  
        att = []  
        for i in range(49):
            ai = -1 * pdist(x1[1][i],x1[2])      
            att.append(ai)
        att = torch.stack(att)
        sft = nn.Softmax(dim=1)
        att_wts = sft(att) #49,16
        d_local = 0
        for i in range(49):
            di = att_wts[i] * pdist(x1[1][i],x2)
            d_local += di  
        return (d_global + d_local)/2

    def train_model(model, optim, epochs=100,margin=0.2):
        data = DataLoaders()
        cat_emb = Category_embedding('data/STL-Dataset/meta_data_final.csv')
        dataloaders = data.dataloaders
        triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=compute_compatability,margin=margin)
        # state=torch.load('/content/drive/MyDrive/datasets/STL-Dataset/attention_model.pth')
        # model.load_state_dict(state['model_state_dict'])
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        # best_model_wts = state['best_weights']
        # best_acc = state['best_acc']  
        best_acc = 0.0
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs))
            print('-' * 10)      
            current_loss = 0.0
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:          
                if phase == 'train':              
                    model.train()  # Set model to training mode
                else:              
                    model.eval()   # Set model to evaluate mode                   

                # Here's where the training happens
                print('Iterating through data...')
                dist_values = []
                for scene,pos,neg,cat in tqdm(dataloaders[phase]):              
                    scene = scene.to(device)
                    pos = pos.to(device)
                    neg = neg.to(device)              
                    emb_list = []
                    for category in cat:
                        key = category.split('|')[-1]
                        emb = embeddings[key]
                        emb_list.append(emb)
                    cat_feat = torch.tensor(emb_list)
                    cat_feat = l2_norm(cat_feat)
                    cat_feat = cat_feat.to(device)              
                    # We need to zero the gradients, don't forget it
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        scene_vfeat,scene_fi = model(scene)
                        pos_vfeat,_ = model(pos)
                        neg_vfeat,_ = model(neg)                                
                        loss = triplet_loss([scene_vfeat,scene_fi,cat_feat],pos_vfeat,neg_vfeat)                
                        # print("Loss : ",loss)                
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        else:                  
                            d_p = compute_compatability([scene_vfeat,scene_fi,cat_feat],pos_vfeat)  
                            d_n = compute_compatability([scene_vfeat,scene_fi,cat_feat],neg_vfeat)
                            dist_values.append(d_p)
                            dist_values.append(d_n)                               
                    current_loss += loss.item() * scene.size(0)                                   
                epoch_loss = current_loss / dataset_sizes[phase]           
                if phase == 'val':
                    dist = torch.stack(dist_values)
                    sft = nn.Softmax(dim=1)
                    dist_probs = sft(dist)
                    epoch_acc = roc_auc_score(gt_val, dist_probs)                                   
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            state={'epoch':epoch,'model_state_dict':model.state_dict(),'model_optim_dict':optimizer.state_dict(),'best_weights':best_model_wts,'best_acc':best_acc} 
            torch.save(state,'attention_model.pth')
        time_since = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_since // 60, time_since % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        # Now we'll load in the best model weights and return it
        model.load_state_dict(best_model_wts)
        self.model = model
    
   


