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




def test_layers(model, test_dataloader):
    layer_mse_model = {}
    layer_mse_prev = {}
    layer_mse_random = {}
    layer_mse_avg_channel = {}
    layer_mse_avg_channel_sequence = {}


    def calculate_baselines_(X_test, layer):
        # Calculate MSE between test data and predictions, excluding the first frame
        mse_prev = np.mean((X_test[:, :-1] - X_test[:, 1:]) ** 2)

        # Calculate MSE for random frame selection baseline, excluding the first frame
        random_indices = np.random.randint(1, X_test.shape[1], size=(X_test.shape[0], X_test.shape[1] - 1))
        X_random = X_test[np.arange(X_test.shape[0])[:, np.newaxis], random_indices]
        mse_random = np.mean((X_test[:, 1:] - X_random) ** 2)

        # Calculate MSE for average pixel value comparison baseline, excluding the first frame
        avg_channel_value = np.mean(X_test[:, 1:], axis=(2, 3))
        X_avg_channel = np.repeat(avg_channel_value[:, :, np.newaxis, np.newaxis], X_test.shape[2], axis=2)
        X_avg_channel = np.repeat(X_avg_channel, X_test.shape[3], axis=3)
        mse_avg_channel = np.mean((X_test[:, 1:] - X_avg_channel) ** 2)

        # Calculate MSE for average pixel value of the entire sequence baseline, excluding the first frame
        avg_channel_value_sequence = np.mean(X_test, axis=(1, 2, 3))
        X_avg_channel_sequence = np.repeat(avg_channel_value_sequence[:, np.newaxis, np.newaxis, np.newaxis], X_test.shape[1] - 1, axis=1)
        X_avg_channel_sequence = np.repeat(X_avg_channel_sequence, X_test.shape[2], axis=2)
        X_avg_channel_sequence = np.repeat(X_avg_channel_sequence, X_test.shape[3], axis=3)
        mse_avg_channel_sequence = np.mean((X_test[:, 1:] - X_avg_channel_sequence) ** 2)

        layer_mse_prev.setdefault(layer, []).append(mse_prev)
        layer_mse_random.setdefault(layer, []).append(mse_random)
        layer_mse_avg_channel.setdefault(layer, []).append(mse_avg_channel)
        layer_mse_avg_channel_sequence.setdefault(layer, []).append(mse_avg_channel_sequence)


    with torch.no_grad():
        for X in test_dataloader:
            layer_activations = model.predict(X, save_act=True)

            for layer in range(model.num_of_layers):
                if layer == 0:
                    # For the bottom layer, compare ahat with the actual image received
                    X_pred = layer_activations[layer]['ahat']
                    X_pred = np.concatenate(X_pred, axis=0)

                    X_actual = X.cpu().numpy()
                    X_actual = np.transpose(X_actual, (0, 1, 4, 2, 3))  # Transpose to match the shape of X_pred

                    # Calculate MSE between predicted and actual image, excluding the first frame
                    mse_model = np.mean((X_actual[0, 1:] - X_pred[:-1]) ** 2)
                else:
                    # For higher layers, compare ahat with the r_up of the layer below
                    X_pred = layer_activations[layer]['ahat']
                    X_pred = np.concatenate(X_pred, axis=0)

                    X_actual = layer_activations[layer - 1]['r_up']
                    X_actual = np.concatenate(X_actual, axis=0)

                    # Apply max pooling to the r_up of the layer below
                    X_actual = np.max(X_actual.reshape(X_actual.shape[0], X_actual.shape[1], X_actual.shape[2] // 2, 2, X_actual.shape[3] // 2, 2), axis=(3, 5))

                    # Calculate MSE between predicted ahat and max-pooled r_up, excluding the first frame
                    mse_model = np.mean((X_actual[1:] - X_pred[:-1]) ** 2)

                layer_mse_model.setdefault(layer, []).append(mse_model)

                # Calculate baselines for each layer
                calculate_baselines_(X_actual, layer)

    for layer in range(model.num_of_layers):
        print(f"Layer {layer}:")
        print("Model MSE: %f" % np.mean(layer_mse_model[layer]))
        print("Previous Timestep MSE: %f" % np.mean(layer_mse_prev[layer]))
        print("Random Timestep MSE: %f" % np.mean(layer_mse_random[layer]))
        print("Average Channel MSE: %f" % np.mean(layer_mse_avg_channel[layer]))
        print("Average Channel Sequence MSE: %f" % np.mean(layer_mse_avg_channel_sequence[layer]))
        print()


