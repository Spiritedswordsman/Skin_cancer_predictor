# Skin_cancer_predictor

### Approach 1 : LSTM

he first LSTM layer is initialized with 128 units and takes as input a sequence of data with a shape defined by the dimensions of X_train. The return_sequences=True parameter ensures that the output of this layer is also a sequence, which is important for feeding the subsequent LSTM layers.
A dropout layer is added after the first LSTM layer with a dropout rate of 0.5. Dropout is a regularization technique that randomly sets a fraction of input units to 0 during training, which helps prevent overfitting and improves generalization.
The second LSTM layer has 64 units and also returns a sequence. It is followed by another dropout layer.
The third LSTM layer has 32 units and does not return a sequence, meaning it produces a single output for the given input sequence.
After the last LSTM layer, a dropout layer is added again to further regularize the model.
Finally, a dense layer with units defined by the number of labels is added, and it uses softmax activation to produce the output probabilities for each label class.


```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 28, 128)           109056    
                                                                 
 dropout (Dropout)           (None, 28, 128)           0         
                                                                 
 lstm_1 (LSTM)               (None, 28, 64)            49408     
                                                                 
 dropout_1 (Dropout)         (None, 28, 64)            0         
                                                                 
 lstm_2 (LSTM)               (None, 32)                12416     
                                                                 
 dropout_2 (Dropout)         (None, 32)                0         
                                                                 
 dense (Dense)               (None, 7)                 231       
                                                                 
=================================================================
Total params: 171,111
Trainable params: 171,111
Non-trainable params: 0
```


Accuracy  -  71.64%


### Approach 2 : CNN

The architecture you provided is a sequential model consisting of several convolutional layers, normalization layers, pooling layers, and dense layers.
The first layer is a convolutional layer with 30 filters of size 5x5, using a stride of 1 and valid padding. It applies the ReLU activation function to introduce non-linearity and extract features from the input images.
The second layer is another convolutional layer with 30 filters of size 3x3, also using a stride of 1 and valid padding. This layer further extracts features from the input images.
A batch normalization layer is added to normalize the activations of the previous layer and improve the training speed and stability.
Next, a max pooling layer with a pool size of 2x2 is used to reduce the spatial dimensions of the feature maps while retaining important features.
The subsequent layers include more convolutional layers, each followed by the ReLU activation function, which help extract higher-level features from the input images.
A group normalization layer with 3 groups is added to normalize the activations of the previous layer.
Another max pooling layer is used to further reduce the spatial dimensions.
A 2x2 convolutional layer with 10 filters and valid padding is added to capture more local features.
The output from the previous layer is flattened to a one-dimensional vector to be fed into the subsequent dense layers.
A normalization layer is added to normalize the activations of the previous layer.
Two dense layers with 256 and 128 units, respectively, and ReLU activation are added to learn complex patterns and representations from the flattened features.
Batch normalization is applied to normalize the activations of the dense layers.
A dropout layer with a dropout rate of 0.1 is included to reduce overfitting by randomly setting a fraction of input units to 0 during training.
Finally, a dense layer with 7 units and softmax activation is added to produce the final probabilities for the 7 output classes.
This architecture combines convolutional and dense layers with normalization, pooling, and dropout techniques to effectively learn and classify features from input images.


```
Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_18 (Conv2D)          (None, 24, 24, 30)        2280      
                                                                 
 conv2d_19 (Conv2D)          (None, 22, 22, 30)        8130      
                                                                 
 batch_normalization_4 (Batc  (None, 22, 22, 30)       120       
 hNormalization)                                                 
                                                                 
 max_pooling2d_5 (MaxPooling  (None, 11, 11, 30)       0         
 2D)                                                             
                                                                 
 conv2d_20 (Conv2D)          (None, 9, 9, 20)          5420      
                                                                 
 conv2d_21 (Conv2D)          (None, 7, 7, 15)          2715      
                                                                 
 conv2d_22 (Conv2D)          (None, 5, 5, 15)          2040      
                                                                 
 group_normalization_1 (Grou  (None, 5, 5, 15)         30        
 pNormalization)                                                 
                                                                 
 max_pooling2d_6 (MaxPooling  (None, 2, 2, 15)         0         
 2D)                                                             
                                                                 
 conv2d_23 (Conv2D)          (None, 1, 1, 10)          610       
                                                                 
 flatten (Flatten)           (None, 10)                0         
                                                                 
 normalization (Normalizatio  (None, 10)               21        
 n)                                                              
                                                                 
 dense (Dense)               (None, 256)               2816      
                                                                 
 batch_normalization_5 (Batc  (None, 256)              1024      
 hNormalization)                                                 
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 7)                 903       
                                                                 
=================================================================
Total params: 59,005
Trainable params: 58,412
Non-trainable params: 593
```

Accuracy – 71.64%
