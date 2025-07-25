import numpy as np
from SAR_utils_new import (Standardize_data, createImageCubes, createImageCubes_cmplx, 
                           get_img_indexes, split_train_test, target, num_classes,
                           AA_andEachClassAccuracy)
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from Load_Data import load_data
from tensorflow.keras import layers
from net_flops import net_flops
import scipy.io as sio
from tensorflow.keras.layers import Conv3D, DepthwiseConv2D
from tensorflow.keras.layers import Dense, Reshape, Input
from tensorflow.keras.models import Model

import tensorflow as tf

from cvnn.layers import ComplexConv3D, complex_input

from Attention_library import CA_2021
from tqdm import tqdm


def predict_by_batching(model, test_idx_list, X_real, X_cmplx, windowSize, batch_size = 500):
    '''
    Function to to perform predictions by dividing large tensor into small ones 
    to reduce load on GPU
    
    Parameters
    ----------
    model: The model itself with pre-trained weights.
    input_tensor_idx: Tensor of diemnsion batches x windowSize x windowSize x channels x 1.
    X_real: Original features descriptors image
    X_cmplx: Original cogerency image
    batch_size: integer value smaller than batches length.

    Returns
    -------
    Predicetd labels
    '''
    num_samples = len(test_idx_list)
    k = 0
    predictions = []
    for i in tqdm(range(0, num_samples, batch_size), desc="Progress"):
        k+=1
        
        batch_real = createImageCubes(X_real, test_idx_list[i:i + batch_size], windowSize)
        batch_cmplx = createImageCubes_cmplx(X_cmplx, test_idx_list[i:i + batch_size], windowSize)

        batch_predictions = model.predict([batch_real, batch_cmplx], verbose=0)
        predictions.append(batch_predictions)
        
    Y_pred_test = np.concatenate(predictions, axis=0)
  
    return Y_pred_test
          
        
        
# Get the data
dataset = 'FL'
window_size = 15
train_per = 0.01
data, T3RF, gt = load_data(dataset)
   

data = Standardize_data(data)
T3RF = Standardize_data(T3RF)


indexes, labels = get_img_indexes(gt, removeZeroindexes = True)


samples_per_class = int(train_per*labels.shape[0]/(np.max(labels)+1))

X_train_idx, y_train, X_test_idx, y_test = split_train_test(indexes, labels, samples_per_class)
X_val_idx, y_val, X_test_idx, y_test = split_train_test(X_test_idx, y_test, samples_per_class)

X_train_cmplx = np.expand_dims(createImageCubes_cmplx(data, X_train_idx, window_size), axis=4)
X_train_real =  np.expand_dims(createImageCubes(T3RF, X_train_idx, window_size), axis=4)

X_val_cmplx = np.expand_dims(createImageCubes_cmplx(data, X_val_idx, window_size), axis=4)
X_val_real =  np.expand_dims(createImageCubes(T3RF, X_val_idx, window_size), axis=4)

class_name = target(dataset)
sample_report = f"{'class': ^25}{'train_num':^10}{'val_num': ^10}{'test_num': ^10}{'total': ^10}\n"
for i in np.unique(gt):
    if i == 0: continue
    sample_report += f"{class_name[i-1]: ^25}{(y_train==i-1).sum(): ^10}{(y_val==i-1).sum(): ^10}{(y_test==i-1).sum(): ^10}{(gt==i).sum(): ^10}\n"
sample_report += f"{'total': ^25}{len(y_train): ^10}{len(y_val): ^10}{len(y_test): ^10}{len(labels): ^10}"
print(sample_report)


   
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
y_val = tf.keras.utils.to_categorical(y_val)


def ddf2Pol(real_x, cmplx_x, num_class, filters = 16):
    # Input Layer
    x_in_r = Input(shape=real_x.shape[1:], name = 'real_x')
    x_in_c = complex_input(shape=cmplx_x.shape[1:], name = 'cmplx_x')
    
    # Real Valued Path
    r = Conv3D(filters, (3,3,3), padding = 'same')(x_in_r)
    r = Conv3D(2*filters, (3,3,3), padding = 'same')(r)
    r = Reshape((r.shape[1], r.shape[2], r.shape[3]*r.shape[4]))(r)
    
    
    # Complex Valued Path
    c = ComplexConv3D(filters, (3,3,3), padding = 'same')(x_in_c)
    c = ComplexConv3D(2*filters, (3,3,3), padding = 'same')(c)
    
    c = tf.concat([tf.math.real(c), tf.math.imag(c)], axis=4)
    c = Reshape((c.shape[1], c.shape[2], c.shape[3]*c.shape[4]))(c)
    
    # Fusion by concatenation    
    f = tf.concat([r,c],axis=3)
    
    # Depthwise Convolution for Spatial feature refinement
    f = DepthwiseConv2D(kernel_size=3)(f)
    
    # Apply Attention
    f = CA_2021(f, reduction=64)
    
    # Global Pooling and Final Classification Block
    p = layers.GlobalAvgPool2D()(f)  # Global pooling in 2D
    
    output_layer = Dense(num_class,activation="softmax")(p)
    
    model=Model(inputs=[x_in_r,x_in_c],outputs=output_layer)
    model.compile(optimizer='ADAM',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model


model = ddf2Pol(X_train_real, X_train_cmplx, num_classes(dataset), filters = 16)
model.summary()

net_flops(model)

# Perform Training
from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='val_accuracy', 
                              patience=10,
                              restore_best_weights=True
                              )


  
    
history = model.fit({'real_x':X_train_real,'cmplx_x': X_train_cmplx}, y_train,
                            batch_size = 64, 
                            verbose = 1, 
                            epochs = 100, 
                            shuffle = True,
                            validation_data = ([X_val_real, X_val_cmplx], y_val),
                            callbacks = [early_stopper] )
    
model.save_weights('./Models_Weights/'+ dataset +'/DDF2Pol.h5')   
    
        
Y_pred_test = predict_by_batching(model, X_test_idx, T3RF, data, window_size, batch_size = 500)

y_pred_test = np.argmax(Y_pred_test, axis=1)
           
        
        
        
kappa = cohen_kappa_score(np.argmax(y_test, axis=1),  y_pred_test)
oa = accuracy_score(np.argmax(y_test, axis=1), y_pred_test)
confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred_test)
each_acc, aa = AA_andEachClassAccuracy(confusion)
        
 
print('OA = ', format(oa*100, ".2f") + " %")
print('AA = ', format(aa*100, ".2f") + " %")
print('Kappa = ', format(kappa*100, ".2f"))

##############################################################################################
def get_class_map(model, X_real, X_cmplx, label, window_size):
    indexes, labels = get_img_indexes(label, removeZeroindexes = False)
    y_pred = predict_by_batching(model, indexes, T3RF, data, window_size, batch_size = 1000)  
    y_pred = (np.argmax(y_pred, axis=1)).astype(np.uint8)
    Y_pred = np.reshape(y_pred, label.shape) + 1
    gt_binary = label.copy()
    gt_binary[gt_binary>0]=1
    return Y_pred*gt_binary


predicted_map = get_class_map(model, T3RF, data, gt, window_size)


Folder = 'Matlab_Outputs/'
Name = 'DDF2Pol'
sio.savemat(Folder + dataset+'/' + Name+'.mat', {Name: predicted_map})
