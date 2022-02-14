#Siamese Network
class SiameseNetwork(nn.Module):
     def __init__(self,embedding_size=128):
          super(SiameseNetwork, self).__init__()
          self.embedding_size = embedding_size
          # Setting up the Sequential of CNN Layers
          self.cnn1 = nn.Sequential( 
          nn.Conv2d(3, 96, kernel_size=11,stride=1),
          nn.ReLU(inplace=True),
          nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
          nn.MaxPool2d(3, stride=2),

          nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
          nn.ReLU(inplace=True),
          nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
          nn.MaxPool2d(3, stride=2),
          nn.Dropout2d(p=0.3),

          nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(3, stride=2),
          nn.Dropout2d(p=0.3),
          )

          # Defining the fully connected layers
          self.fc1 = nn.Sequential(
            nn.Linear(256*25*25, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.embedding_size,self.embedding_size))
    
     def freeze_params(self):
      for name,param in self.named_parameters():
        if param.requires_grad and "fc" not in name:
          param.requires_grad = False

     def forward(self, x):
          # Forward pass 
          output = self.cnn1(x)          
          output = output.view(output.size()[0], -1)          
          output = self.fc1(output)
          # print(output.size())
          return output