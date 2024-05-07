import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


num_epochs = 30
lr = 0.001
nt = 10 # num of time steps
batch_size = 8
layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.]]).cuda())

train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

kitti_train = KITTI(train_file, train_sources, nt)
kitti_val = KITTI(val_file, val_sources, nt)

train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(kitti_val, batch_size=batch_size, shuffle=False)

stack1=60
stack2=120
R_stack3=240
hidden_sizes = (3, stack1, stack2)
r_hidden_sizes = (stack1, stack2, R_stack3)

precnet = PrecNetModel(hidden_sizes,r_hidden_sizes)

def train_one_epoch(model, data_loader, optimizer, loss_fn):
  running_loss = 0.0
  pbar = tqdm(data_loader, position = 0, leave= True)
  for i, seq in enumerate(pbar):
      loss = model.train_step(seq, optimizer, loss_fn)
      running_loss += loss.item()
      avg_loss = running_loss / (i + 1)
      pbar.set_postfix(loss=avg_loss)

  return running_loss / len(data_loader)

def val_one_epoch(model, data_loader, loss_fn):
  running_loss = 0.0
  with torch.no_grad():
    for i, seq in enumerate(data_loader):
        loss = model.test_step(seq, loss_fn)
        running_loss += loss

  return running_loss/len(data_loader)

def train_prec(model, num_epochs, train_loader, val_loader):
    lr_decay_epoch = num_epochs // 2
    initial_lr = 0.001
    if torch.cuda.is_available():
      print('Using GPU.')
      model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = initial_lr)
    scheduler = StepLR(optimizer, step_size=lr_decay_epoch, gamma=0.1)
    loss_fn = nn.L1Loss()
    print('Run for', num_epochs, 'epochs')

    for epoch in range(0, num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = val_one_epoch(model, train_loader, loss_fn)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{num_epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - lr: {current_lr}')
