import numpy as np

def test_mse(model, test_dataloader):
    X_test = []
    X_hat = []
    with torch.no_grad():
        for X in test_dataloader:
            X_test.append(X.cpu().numpy())
            X_hat.append(model.predict(X).cpu().numpy())

    X_test = np.concatenate(X_test, axis=0)
    X_hat = np.concatenate(X_hat, axis=0)

    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

    # Calculate MSE between test data and predictions, excluding the first frame
    mse_model = np.mean((X_test[:, 1:] - X_hat[:, 1:]) ** 2)
    mse_prev = np.mean((X_test[:, :-1] - X_test[:, 1:]) ** 2)

    # Calculate MSE for random frame selection baseline, excluding the first frame
    random_indices = np.random.randint(1, X_test.shape[1], size=(X_test.shape[0], X_test.shape[1] - 1))
    X_random = X_test[np.arange(X_test.shape[0])[:, np.newaxis], random_indices]
    mse_random = np.mean((X_test[:, 1:] - X_random) ** 2)

    # Calculate MSE for average pixel value comparison baseline, excluding the first frame
    avg_pixel_value = np.mean(X_test[:, 1:], axis=(2, 3))
    X_avg_pixel = np.repeat(avg_pixel_value[:, :, np.newaxis, np.newaxis], X_test.shape[2], axis=2)
    X_avg_pixel = np.repeat(X_avg_pixel, X_test.shape[3], axis=3)
    mse_avg_pixel = np.mean((X_test[:, 1:] - X_avg_pixel) ** 2)

    # Calculate MSE for average pixel value of the entire sequence baseline, excluding the first frame
    avg_pixel_value_sequence = np.mean(X_test, axis=(1, 2, 3))
    X_avg_pixel_sequence = np.repeat(avg_pixel_value_sequence[:, np.newaxis, np.newaxis, np.newaxis], X_test.shape[1] - 1, axis=1)
    X_avg_pixel_sequence = np.repeat(X_avg_pixel_sequence, X_test.shape[2], axis=2)
    X_avg_pixel_sequence = np.repeat(X_avg_pixel_sequence, X_test.shape[3], axis=3)
    mse_avg_pixel_sequence = np.mean((X_test[:, 1:] - X_avg_pixel_sequence) ** 2)

    print("Previous fixation MSE: %f" % mse_prev)
    print("Model MSE: %f" % mse_model)
    print("Random Frame Selection MSE (excluding first frame): %f" % mse_random)
    print("Average Pixel Value (of target): %f" % mse_avg_pixel)
    print("Average Pixel Value (of entire sequence): %f" % mse_avg_pixel_sequence)

