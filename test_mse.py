import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')
test_dataset = KITTI(test_file, test_sources, nt)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def test(model, test_dataloader):
    if torch.cuda.is_available():
      print('Using GPU.')
      model.cuda()
    X_test = []
    X_hat = []
    with torch.no_grad():
        for X in test_dataloader:
            X_test.append(X.cpu().numpy())
            X_hat.append(model.predict(X).cpu().numpy())

    X_test = np.concatenate(X_test, axis=0)
    X_hat = np.concatenate(X_hat, axis=0)

    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

    # Calculate MSE of PredNet predictions
    mse_model = np.mean((X_test[:, 1:] - X_hat[:, 1:]) ** 2)  # look at all timesteps except the first
    mse_prev = np.mean((X_test[:, :-1] - X_test[:, 1:]) ** 2)


    print(f"Model MSE: {mse_model:.5f}")
    print(f"Previous Frame MSE: {mse_prev:.5f}")

