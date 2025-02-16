## Pedal Detection

### Train the model

```
python train.py
```

### Inference

Run the following command will inference the model and save the results in .npy files.
```
python inference_new.py
```

### View the result

This will read the .npy files saved in inference stage and calculate all metrics (MSE, MAE, F1) and plot the results.
```
python calculate_metric.py
```