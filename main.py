from data import get_dataloaders
from models import RNN, TCN
from train import train_and_validate

from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch import device, cuda
dev = device('cuda:0' if cuda.is_available() else 'cpu')

NR_CLASSES = 4

### Data loading in batchs ###
BATCH_SIZE = 2
intra_loader_train, intra_loader_test = get_dataloaders(type="intra-subject", batch_size=BATCH_SIZE)
cross_loader_train, cross_loader_test = get_dataloaders(type="cross-subject", batch_size=BATCH_SIZE)

# General hyperparameters
LEARNING_RATE= 0.1
NR_EPOCHS = 3
NR_LAYERS = 1

rnn = RNN(input_size=248, hidden_size=248, num_layers=NR_LAYERS, output_size=NR_CLASSES).to(dev)
cross_entropy = CrossEntropyLoss()
#optimizer = Adam(rnn.parameters(), lr=LEARNING_RATE)
optimizer = SGD(rnn.parameters(), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer, gamma=0.9)

""" train_and_validate(criterion=cross_entropy, optimizer=optimizer, model=rnn, scheduler=scheduler, 
                   dataloader_train=intra_loader_train, dataloader_test=intra_loader_test, epochs=NR_EPOCHS) """


HIDDEN_UNITS_PER_LAYER = 25
NR_LEVELS = 8
dropout = 0.05

tcn = TCN(input_size=248, output_size=NR_CLASSES, num_channels=[HIDDEN_UNITS_PER_LAYER]*NR_LEVELS, kernel_size=2, dropout=0.2).to(dev)
cross_entropy = CrossEntropyLoss()
optimizer = SGD(tcn.parameters(), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer, gamma=0.9)

train_and_validate(criterion=cross_entropy, optimizer=optimizer, model=tcn, scheduler=scheduler, 
                   dataloader_train=cross_loader_train, dataloader_test=cross_loader_test, epochs=NR_EPOCHS)