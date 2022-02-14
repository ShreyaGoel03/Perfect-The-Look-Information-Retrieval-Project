#Resnet with Attention
class Model(nn.Module):
  def __init__(self, embedding_size=128):
    super(Model, self).__init__()      
    # self.weight = nn.Parameter(torch.Tensor([0]))         
    self.embedding_size = embedding_size
    self.model = models.wide_resnet50_2(pretrained=True)
    self.num_ftrs = self.model.fc.in_features            
    self.model.fc = nn.Sequential(
            nn.Linear(self.num_ftrs, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.embedding_size,self.embedding_size)         
        )
    self.model.fc1 = nn.Sequential( #1024*7*7
            nn.Linear(1024, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.embedding_size,self.embedding_size)         
        )
  
  def l2_norm(self,input):
      input_size = input.size()
      buffer = torch.pow(input, 2)      
      normp = torch.sum(buffer, 1).add_(1e-10)
      norm = torch.sqrt(normp)

      _output = torch.div(input, norm.view(-1, 1).expand_as(input))

      output = _output.view(input_size)

      return output

  def freeze_params(self):
    for name,param in self.model.named_parameters():
      if param.requires_grad and "fc" not in name:
        param.requires_grad = False

  def forward(self,x):
    x = self.model.conv1(x)
    x = self.model.bn1(x)
    x = self.model.relu(x)
    x = self.model.maxpool(x)
    x = self.model.layer1(x)
    x = self.model.layer2(x)
    x = self.model.layer3(x)
    temp = x
    x = self.model.layer4[0].conv1(x)    
    x = self.model.layer4[0].bn1(x)
    x = self.model.layer4[0].conv2(x)
    feat_temp = x             
    x = self.model.layer4(temp)    
    x = self.model.avgpool(x)        
    x = torch.flatten(x, 1)    
    feat_temp = torch.flatten(feat_temp, 2)      
    f_i = []
    for i in range(49):
      y = self.model.fc1(feat_temp[:,:,i])
      feat_map = self.l2_norm(y)
      f_i.append(feat_map)                       
    x = self.model.fc(x) 
    self.feat_map = torch.stack(f_i)
    self.visual_features = self.l2_norm(x) 
    # print(self.feat_map.size(),self.visual_features.size())   
    return self.visual_features, self.feat_map