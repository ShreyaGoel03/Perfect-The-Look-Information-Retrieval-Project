from train.py import *
from models.attention.py import *
from create_dataloader.py import *

def test_model(model):  
    dist_values = []
    data = DataLoaders()
    cat_emb = Category_embedding('data/STL-Dataset/meta_data_final.csv')
    dataloaders = data.dataloaders
    dataloader_test = dataloaders['test']
    with torch.no_grad():
        for scene,prod1,prod2,cat in tqdm(dataloader_test):    
            scene = scene.to(device)
            prod1 = prod1.to(device)
            prod2 = prod2.to(device) 
            scene_vfeat,scene_fi = model(scene)    
            prod1_vfeat,_ = model(prod1)    
            prod2_vfeat,_ = model(prod2) 
            emb_list = []
            for category in cat:
            key = category.split('|')[-1]
            emb = embeddings[key]
            emb_list.append(emb)
            cat_feat = torch.tensor(emb_list)
            cat_feat = l2_norm(cat_feat)
            cat_feat = cat_feat.to(device)   
            d_1 = compute_compatability([scene_vfeat,scene_fi,cat_feat],prod1_vfeat)  
            d_2 = compute_compatability([scene_vfeat,scene_fi,cat_feat],prod2_vfeat)
            dist_values.append(d_1)
            dist_values.append(d_2)                                     
    dist = torch.stack(dist_values)
    sft = nn.Softmax(dim=1)
    dist_probs = sft(dist)
    auc = roc_auc_score(gt_test, dist_probs) 
    return auc

if __name__ == "__main__":
    model = Model()
    model.freeze_params()
    gt_val = []
    data = DataLoaders()
    dataset_validate = data.dataloaders['val']
    for i in range(len(dataset_validate)):
        gt_val.append(1)
        gt_val.append(0)
    gt_val = np.array(gt_val)
    # state=torch.load('attention_model.pth')
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    # optimizer.load_state_dict(state['model_optim_dict'])
    train_obj = Train()    
    train_obj.train_model(model,optimizer,1)
    model = train_obj.model
    auc = test_model(model)    
    print("AUC: ",auc)



