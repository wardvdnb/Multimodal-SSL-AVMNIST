## Best DINO Model Performances (Multimodal and Unimodal) [Step 1]
The first step is a 100 epoch run on 3 different seeds that calculates the mean accuracy on an MLP probe and KNN accuracy for all architectures, the hyperparameters are kept static for all models. With this initial augmentation setup, the experiments suggest mlp probe metric is probably best.

### Multimodal

1. **Dual VIT**

- #### Run: multi_dual_vit_train_loss_12062025_011930
- Best Mean Accuracy: 65.2833
- MLP Accuracy: 65.2833 ± 0.1159
- k-NN Accuracy: 61.0000 ± 0.0707

- #### Run: multi_dual_vit_mlp_acc_13062025_072024
- Best Mean Accuracy: 65.1000
- MLP Accuracy: 65.1000 ± 0.8193
- k-NN Accuracy: 60.7500 ± 0.7365

2. **CentralNet CNNs**

- #### Run: multi_central_mlp_acc_13062025_102517
- Best Mean Accuracy: 63.5367
- MLP Accuracy: 63.5367 ± 0.2451
- k-NN Accuracy: 60.0900 ± 0.0000

- #### Run: multi_central_train_loss_12062025_033521
- Best Mean Accuracy: 12.6267
- MLP Accuracy: 12.6267 ± 0.0368
- k-NN Accuracy: 11.5400 ± 1.0737

3. **Cross attention CNNs**

- #### Run: multi_cross_attention_train_loss_12062025_032057
- Best Mean Accuracy: 52.0067
- MLP Accuracy: 50.5600 ± 0.1883
- k-NN Accuracy: 52.0067 ± 0.5280

- #### Run: multi_cross_attention_mlp_acc_13062025_102431
- Best Mean Accuracy: 49.4667
- MLP Accuracy: 49.4667 ± 0.7878
- k-NN Accuracy: 48.8567 ± 2.5101


### Unimodal Image
1. **Simple CNN** (only one trained)

- #### Run: image_simple_train_loss_12062025_053829
- Best Mean Accuracy: 64.1433
- MLP Accuracy: 64.1433 ± 0.0499
- k-NN Accuracy: 61.3700 ± 0.0990

- #### Run: image_simple_mlp_acc_13062025_103957
- Best Mean Accuracy: 63.8367
- MLP Accuracy: 63.8367 ± 0.2485
- k-NN Accuracy: 61.3267 ± 0.3418


### Unimodal Spectrogram

1. **Central CNN**

- #### Run: spectrogram_central_mlp_acc_13062025_110513
- Best Mean Accuracy: 39.8467
- MLP Accuracy: 39.8467 ± 0.1367
- k-NN Accuracy: 14.2700 ± 0.0000

- #### Run: spectrogram_central_train_loss_12062025_063051
- Best Mean Accuracy: 31.0467
- MLP Accuracy: 31.0467 ± 0.1466
- k-NN Accuracy: 14.3200 ± 0.0000

2. **Simple CNN**
- #### Run: spectrogram_simple_train_loss_12062025_061152
- Best Mean Accuracy: 26.6967
- MLP Accuracy: 26.6967 ± 0.4071
- k-NN Accuracy: 15.4900 ± 0.3476

- #### Run: spectrogram_simple_train_loss_13062025_185325
- Best Mean Accuracy: 26.6967
- MLP Accuracy: 26.6967 ± 0.4071
- k-NN Accuracy: 15.4900 ± 0.3476

- #### Run: spectrogram_simple_mlp_acc_13062025_110119
- Best Mean Accuracy: 26.3800
- MLP Accuracy: 26.3800 ± 0.8592
- k-NN Accuracy: 15.0400 ± 0.5243

3. **CNN + LSTM**

- #### Run: spectrogram_lstm_train_loss_12062025_071132
- Best Mean Accuracy: 25.4567
- MLP Accuracy: 25.4567 ± 0.1713
- k-NN Accuracy: 14.9800 ± 0.0000

- #### Run: spectrogram_lstm_mlp_acc_13062025_110515
- Best Mean Accuracy: 25.1733
- MLP Accuracy: 25.1733 ± 0.1969
- k-NN Accuracy: 15.0900 ± 0.1131

---

### Remaining models

- #### Run: multi_vit_mlp_acc_12062025_164126
  Best Mean Accuracy: 54.0000
  MLP Accuracy: 48.5167 ± 0.1782
  k-NN Accuracy: 54.0000 ± 0.0000

- #### Run: multi_vit_train_loss_12062025_000305
  Best Mean Accuracy: 53.2233
  MLP Accuracy: 44.7100 ± 0.6828
  k-NN Accuracy: 53.2233 ± 0.1650

- #### Run: multi_simple_train_loss_11062025_224944
  Best Mean Accuracy: 50.6800
  MLP Accuracy: 50.6800 ± 1.3306
  k-NN Accuracy: 47.9433 ± 0.5893

- #### Run: multi_resnet_train_loss_12062025_031858
  Best Mean Accuracy: 49.9967
  MLP Accuracy: 44.2800 ± 1.4596
  k-NN Accuracy: 49.9967 ± 1.2940

- #### Run: multi_simple_gated_train_loss_11062025_231213
  Best Mean Accuracy: 49.5600
  MLP Accuracy: 49.5600 ± 1.8805
  k-NN Accuracy: 48.7167 ± 0.9944

- #### Run: multi_simple_gated_mlp_acc_12062025_163906
  Best Mean Accuracy: 49.5267
  MLP Accuracy: 49.5267 ± 0.2945
  k-NN Accuracy: 45.3500 ± 0.0000

- #### Run: multi_simple_mlp_acc_12062025_144501
  Best Mean Accuracy: 49.4267
  MLP Accuracy: 49.4267 ± 0.5222
  k-NN Accuracy: 47.2300 ± 0.0283

- #### Run: multi_resnet_mlp_acc_13062025_080649
  Best Mean Accuracy: 49.0100
  MLP Accuracy: 45.4933 ± 0.1586
  k-NN Accuracy: 49.0100 ± 0.0000

- #### Run: multi_lstm_train_loss_11062025_233950
  Best Mean Accuracy: 48.6933
  MLP Accuracy: 40.6567 ± 1.1501
  k-NN Accuracy: 48.6933 ± 1.6837

- #### Run: multi_lstm_mlp_acc_12062025_163902
  Best Mean Accuracy: 47.7000
  MLP Accuracy: 40.5267 ± 0.2100
  k-NN Accuracy: 47.7000 ± 0.0000

- #### Run: multi_mobile_vit_train_loss_12062025_030854
  Best Mean Accuracy: 43.5100
  MLP Accuracy: 23.7233 ± 2.6215
  k-NN Accuracy: 43.5100 ± 3.7123

- #### Run: multi_mobile_vit_mlp_acc_13062025_072506
  Best Mean Accuracy: 33.7100
  MLP Accuracy: 23.9533 ± 0.1342
  k-NN Accuracy: 33.7100 ± 0.0000

- #### Run: spectrogram_vit_mlp_acc_13062025_110518
  Best Mean Accuracy: 21.7800
  MLP Accuracy: 21.7800 ± 1.5432
  k-NN Accuracy: 13.7467 ± 0.5421

- #### Run: spectrogram_vit__train_loss_21062025_170025
  Best Mean Accuracy: 20.7133
  MLP Accuracy: 20.7133 ± 0.4259
  k-NN Accuracy: 13.5733 ± 0.6175

- #### Run: spectrogram_resnet_train_loss_12062025_090922
  Best Mean Accuracy: 16.5733
  MLP Accuracy: 16.5733 ± 0.3016
  k-NN Accuracy: 13.6533 ± 0.1461

- #### Run: spectrogram_resnet_mlp_acc_13062025_110515
  Best Mean Accuracy: 16.5567
  MLP Accuracy: 16.5567 ± 0.3120
  k-NN Accuracy: 13.4333 ± 0.0660

- #### Run: spectrogram_mobile_vit_mlp_acc_13062025_110517
  Best Mean Accuracy: 15.8300
  MLP Accuracy: 15.8300 ± 0.2083
  k-NN Accuracy: 11.8033 ± 1.0229

- #### Run: spectrogram_mobile_vit_train_loss_12062025_104902
  Best Mean Accuracy: 15.3000
  MLP Accuracy: 15.3000 ± 0.3677
  k-NN Accuracy: 12.5633 ± 0.0850

- #### Run: multi_central_mse_train_loss_20062025_225356
  Best Mean Accuracy: 13.6067
  MLP Accuracy: 13.6067 ± 0.3210
  k-NN Accuracy: 13.2567 ± 1.4459

- #### Run: multi_central_semi_supervised_train_loss_20062025_225312
  Best Mean Accuracy: 13.2367
  MLP Accuracy: 13.2367 ± 0.0205
  k-NN Accuracy: 12.9300 ± 0.5564

- #### Run: multi_central_infonce_train_loss_20062025_225328
  Best Mean Accuracy: 12.6733
  MLP Accuracy: 12.6733 ± 0.0759
  k-NN Accuracy: 12.3733 ± 0.4903