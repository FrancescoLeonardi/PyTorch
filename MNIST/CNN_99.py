import time
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# CLASS
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.dropout1D = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(1024, 10)
        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.max_pool2d(F.selu(self.conv2(F.selu(self.conv1(x)))), 2) 
        x = self.norm1(x)
        x = F.max_pool2d(F.selu(self.conv4(F.selu(self.conv3(x)))), 2)
        x = self.norm2(x)    
        
        x = self.conv5(x)
        x = self.norm3(x)    
        
        x = x.view(-1, 1024)
        x = self.dropout1D(x)
        
        x = self.fc1(x)
        
        return F.log_softmax(x, dim=1)
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# HYPERPARAMETERS
n_epochs = 4
training_batch = 100
testing_batch = 1000

learning_rate = 0.005
momentum = 0.95
batch_print = 10


# SEED
#seed = 0
#torch.manual_seed(seed)
# SEED 0 -> 99.00
# SEED 1 -> 99.09
# SEED 2 -> 99.10
# RANDOM -> ?

# DATA PREPROCESSING
transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train = True, download = True, 
                             transform = transform), batch_size = training_batch, shuffle = True)
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train = False, download = True,
                             transform = transform), batch_size = testing_batch, shuffle = True)

 
    
# CREATE CNN
cnn = CNN()
cnn.to(device)
optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=momentum)
    
    
# TRAINING
def training(epoch, verbose = True):
    if verbose:
        print("")
    start = time.time()
    cnn.train()
    for batch, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = cnn(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if verbose:
            if batch%batch_print==0:        
                print('Training Epoch: {} [{:.0f}%]\t\tLoss: {:.3f}\t\tTime: {:.2f} seconds'
                      .format( epoch, 100. * batch / len(train_loader), loss.item(), time.time()-start))
    if verbose:
        print("")
        
# TESTING 
def test(epoch):
    start = time.time()
    cnn.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = cnn(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('Test set: {}'.format(epoch))
    print('Avg loss: {:.3f}\t\tAccuracy: {}/{}\t\tTime: {:.2f} seconds'
          .format(test_loss,  correct, len(test_loader.dataset), time.time()-start))
  
        
    
# RUN CNN  
for epoch in range(1, n_epochs + 1):
    training(epoch, verbose=True)
    test(epoch)    