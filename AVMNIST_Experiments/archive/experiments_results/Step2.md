## Best DINO Model Performances (Multimodal and Unimodal) [Step 2]
The second step is a 100 epoch run on 3 different seeds that calculates the mean accuracy on an MLP probe and KNN accuracy for all architectures, the hyperparameters are tuned this time, over 50 trials with 50 epochs each.
This time every model was tracked using mlp_acc only (linear probe). We notice that using the (unimodal) centralnet architectures as encoders for our models perform best, in addition to the hypertuned parameters performing worse than the previous static ones. The optimal set of hyperparameters thus remains the one in ``config_multimodal_dino_old_augments.yaml``.

### Multimodal

- #### Run: Subfolder: multi_central_mlp_acc_24062025_135106
  - Best Mean Accuracy: 60.4700
  - MLP Accuracy: 60.4700 ± 1.0453
  - k-NN Accuracy: 57.8767 ± 0.5704

- #### Run: multi_dual_vit_mlp_acc_24062025_135108
  - Best Mean Accuracy: 57.4467
  - MLP Accuracy: 53.8133 ± 0.7641
  - k-NN Accuracy: 57.4467 ± 0.8155

- #### Run: multi_cross_attention_mlp_acc_24062025_135106
  - Best Mean Accuracy: 55.3500
  - MLP Accuracy: 55.3500 ± 0.7809
  - k-NN Accuracy: 54.9633 ± 0.7829


### Unimodal Image
1. **Simple CNN** (only one trained)

- #### Run:image_simple_mlp_acc_24062025_135023
  - Best Mean Accuracy: 64.4067
  - MLP Accuracy: 64.4067 ± 0.2951
  - k-NN Accuracy: 61.2467 ± 0.1592



### Unimodal Spectrogram

1. **Central CNN**

- #### Run: spectrogram_central_mlp_acc_24062025_135023
  - Best Mean Accuracy: 35.4400
  - MLP Accuracy: 35.4400 ± 0.1257
  - k-NN Accuracy: 16.2600 ± 0.0000

- #### Run: spectrogram_lstm_mlp_acc_24062025_135028
  - Best Mean Accuracy: 27.3967
  - MLP Accuracy: 27.3967 ± 1.0867
  - k-NN Accuracy: 18.3167 ± 0.4855

- #### Run: spectrogram_simple_mlp_acc_24062025_135027
  - Best Mean Accuracy: 25.3833
  - MLP Accuracy: 25.3833 ± 0.6463
  - k-NN Accuracy: 14.3967 ± 0.3064