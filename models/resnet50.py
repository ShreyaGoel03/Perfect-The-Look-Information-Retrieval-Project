
#Resnet baseline
class Baseline_Model(nn.Module):
  def __init__(self, embedding_size=128):
    super(Baseline_Model, self).__init__()                       
    self.model = models.wide_resnet50_2(pretrained=True)           

  def forward(self,x):
    x = self.model.conv1(x)
    x = self.model.bn1(x)
    x = self.model.relu(x)
    x = self.model.maxpool(x)
    x = self.model.layer1(x)
    x = self.model.layer2(x)
    x = self.model.layer3(x)
    x = self.model.layer4(x)    
    x = self.model.avgpool(x)    
    # print(x.size())
    x = torch.flatten(x, 1)
    return x