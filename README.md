# Epistemic Uncertainty (Model-Related)
Arises from limitations in the model
# Quantifying-uncertainty-in-1D-using-neural-networks
 The approach involves creating a fully connected neural network with one hidden layer containing 64 neurons. The model is trained to minimize the mean squared error (MSE) loss. After training, the model is used to predict the output, and the residuals are calculated. Since the model is trained using 50 independent seeds (due to random weight initialization), the residuals will be similar but not identical across different seeds.  From the variation of residual, we find the standard deviation at every location. Finally, we get the uncertainty at every location.

# why it's crucial to Quantifying-uncertainty?
The model provides predictions at various locations, but without uncertainty, we cannot judge how reliable these predictions are. By calculating the uncertainty at every location, we gain insights into the confidence of the model.

# Why train the model with different seeds?
Training the model with 50 independent seeds introduces slight variations in the learned parameters due to random weight initialization. The resulting residual variation reflects the sensitivity of the model to different training configurations. Quantifying this variation helps capture how robust the model's predictions are under these variations.

# Interpreting Uncertainty

The model predicts varying uncertainty at different locations. High uncertainty at specific locations may indicate areas where the model struggles, such as regions with sparse or noisy data.

