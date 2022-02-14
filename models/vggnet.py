
#VGG Net
class VGG_model(nn.Module):
  def __init__(self, embedding_size=128):
    super(VGG_model, self).__init__()
    self.model = models.vgg19(pretrained = True)

  def forward(self, x):
    x = self.model.features(x)
    x = self.model.avgpool(x)
    x = torch.flatten(x,1)
    x = self.model.classifier(x)
    return x