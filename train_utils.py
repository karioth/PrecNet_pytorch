import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

def train_one_epoch(model, data_loader, optimizer, loss_fn, layer_loss_weights):
  running_loss = 0.0
  pbar = tqdm(data_loader, position = 0, leave= True)
  for i, seq in enumerate(pbar):
      loss = model.train_step(seq, optimizer, loss_fn, layer_loss_weights)
      running_loss += loss.item()
      avg_loss = running_loss / (i + 1)
      pbar.set_postfix(loss=avg_loss)

  return running_loss / len(data_loader)

def val_one_epoch(model, data_loader, loss_fn, layer_loss_weights):
  running_loss = 0.0
  with torch.no_grad():
    for i, seq in enumerate(data_loader):
        loss = model.test_step(seq, loss_fn, layer_loss_weights)
        running_loss += loss

  return running_loss/len(data_loader)

def train_prec(model, num_epochs, train_loader, val_loader, layer_loss_weights):
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
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, layer_loss_weights)
        val_loss = val_one_epoch(model, val_loader, loss_fn, layer_loss_weights)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{num_epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - lr: {current_lr}')
