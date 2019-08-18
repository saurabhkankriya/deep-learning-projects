# Deep-Learning-projects

## Different MLP architecture implementation on MNIST using keras
**Objective:** To build different MLP architectures on the MNIST dataset by experimenting with different  weight initializers, number of hidden layers, Optimizers, BatchNorm, Dropout etc.

1. Experiemented different MLP architectures on the MNIST dataset:

    * 2 hidden layers - 784 (input) - 256 - 64 - 10 (ouput)
    * 3 hidden layers - 784 (input) - 128 - 64 - 32- 10 (output)
    * 5 hidden layers - 784 (input) - 512 - 256 - 128 - 64 - 32 - 10 (output)

2. Initialized diffferent weight vectors using:

	* glorot-normal
    * glorot-uniform
    * he-normal
    * he-uniform
	* random_normal
	* random_uniform
	
3. For every architecture, plotted epoch vs loss for training and validation data.

4. For sanity check, plotted violin plots of weights after training the model.

5. Also, performed batch normalization and dropout and it resulted in increase in the accuracy.

6. Conducted a comparison to see what performs better: normbatch before dropout vs dropout before normbatch. Found out dropout before normbatch performed slightly better by looking at the test accuracy.

## Different CNN architecture implementation on MNIST using keras
**Objective:** To build different CNN architectures on the MNIST dataset by experimenting with different kernel sizes, Conv2D layers, BatchNorm, Dropout etc

1. Created 3 different models of kernel size: [2 2], [5 5], [7  7]
2. Tried different number of conv layer, max pooling layers, dropout rates and optimizer.
3. Plotted error plot of number of epochs against training and validataion set.
Model details
    * **Model 1:**
     kernel [2*2]
     strides= (1, 1)
     3 Conv2D layers followed by 2 MaxPool layers of size (2,2)
     3 hidden layers
    * **Model 2:**
     kernel [5*5]
     strides= (2, 2)
     padding = 'same'
     maxpoolsize= (4,4)
     optimizer = 'adam'
     conv2d -> dense(512) -> conv2d-maxpool-dropout-flatten -> dense(256) ->      dense(128)
     * **Model 3:**
      kernel [7*7]
      Test score: 0.030372555697989446
      Test accuracy: 0.9902kernel [7*7]
      strides= (3, 3)
      optimizer = 'RMSprop'
      used BatchNorm
      * **Model 4:**
      kernel [2*2]
      strides= (1, 1)
      3 Conv2D layers followed by 2 MaxPool layers of size (2,2)
      3 hidden layers
      kernel_initializer = 'lecun_normal'
      activation='tanh'
      optimizer = 'sgd'
      * **Model 5:**
      kernel [5*5]
      strides= (2, 2)
      padding = 'same'
      maxpoolsize= (4,4)
      optimizer = 'adam'
      conv2d -> dense(512) -> conv2d-maxpool-dropout-flatten -> dense(256) -> dense(128)
      kernel_initializer = 'he_uniform'
      activation_function = 'elu'
      optimizer = 'nadam'
      * **Model 6:**
      kernel [7*7]
      strides= (3, 3)
      optimizer = 'RMSprop'
      used BatchNorm
      kernel_initializer = 'glorot_normal'
      optimizer='adamax'
      activation='selu

## LSTM on Amazon Fine Food Reviews
**Objective:** To build different RNN architectures of LSTM on the Amazon Fine Food dataset by experimenting with different weight initializers, LSTM gates, number of hidden layers, Optimizers, BatchNorm, Dropout etc and predict whether a given review is positive or negative. 

1. Experiement with 3 models; each had different architecture, weight initializers, activation function, number of hidden layers, number of LSTM layers, optimization function etc.
2. Also made use of different dropout values and batchnormalization of the above architectures.
3. Plotted error plot of number of epochs against training and validataion set.
5. The number of epochs considered: 15.
    * **Model 1**
    Embedding-LSTM(100)-Dense(1)
    batchsize = 5000
    * **Model 2**
    Embedding-LSTM(128)-Dense(512)+Dropout(0.25)-LSTM(64)-Dense(256)-Dense(1)
    batchsize = 1000
    * **Model 3**
    Embedding-LSTM(100)-Dense(128)+Dropout(0.25)+BatchNorm-LSTM(100)-Dense(64)+    Dropout(0.5)+BatchNorm-LSTM(100)-
    Dense(32)-Dense(16)+BatchNorm-LSTM(64)-Dense(16)+BatchNorm-Dense(1)
    batchsize = 1500




