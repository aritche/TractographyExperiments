import torch
from torchsummary import summary
#from pytorch_modelsize import SizeEstimator

class CustomNetwork(torch.nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()

        # INPUT: 3D volume (144 x 144 x 144 x 3)
        self.conv1 = torch.nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(9,9,9), stride=2, padding=3)
        self.conv2 = torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(9,9,9), stride=2, padding=3)
        self.conv3 = torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(9,9,9), stride=2, padding=3)
        self.conv4 = torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(9,9,9), stride=2, padding=3)
        self.maxPool = torch.nn.MaxPool3d(kernel_size=8)
        self.relu  = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.maxPool(x)

        return x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data_files = [torch.randn(3, 144, 144, 144, device='cuda') for i in range(105)]

    def __getitem__(self, idx):
        return [self.data_files[idx], torch.randn(256, 1, 1, 1, device='cuda')]

    def __len__(self):
        return len(self.data_files)

BATCH_SIZE = 2
train_dataset = CustomDataset()
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

torch.cuda.empty_cache()
model = CustomNetwork().to('cuda')
summary(model, (3, 144, 144, 144))

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
for epoch in range(10):
    print("Epoch %d/%d" % (epoch, 10))
    train_loss = 0
    num_batch = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()
        output = model.forward(inputs)

        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss
        num_batch += 1

    print("Mean batch loss: %.3f" % (train_loss / num_batch))
