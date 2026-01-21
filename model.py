import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class CNN(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3)

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(128 * 14 * 14, 128)
        
        self.fc2 = nn.Linear(128,6)

    def forward(self, x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x



model = CNN()
load_model = torch.load('CNN_model.pth',map_location=torch.device('cpu'))

model.load_state_dict(load_model)
model.eval()


transform = transforms.Compose(
                    [transforms.Resize((128,128)),
                    transforms.ToTensor()]
                    )

classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def predict(img):
    X = transform(img).unsqueeze(0)
    with torch.no_grad():
        ouput = model(X)
        ouput = torch.softmax(ouput,dim=1)
        prob, pred = torch.max(ouput,dim=1)

    return classes[pred.item()], prob.item(), ouput

