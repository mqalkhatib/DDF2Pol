# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:21:37 2022

@author: malkhatib
"""
from sklearn.model_selection import train_test_split
import numpy as np
from operator import truediv
import random 
from sklearn.utils import shuffle

###########################################################################################
def get_img_indexes (class_map, removeZeroindexes = True):
    """
    Get indices of elements in the class map.
    
    Parameters:
    class_map (numpy array): The class map (2D array).
    removeZero (bool): If True, return indices of non-zero elements, 
                       otherwise return indices of all elements.
    
    Returns:
    tuple: (indices, labels)
           - indices: List of tuples representing the indices of the selected elements.
           - labels: Array of labels corresponding to the indices.
    """
    if removeZeroindexes:
        # Get indices of non-zero values
        indices = np.argwhere(class_map != 0)
    else:
        # Get indices of all elements (including zeros)
        indices = np.argwhere(class_map != None)
    
    # Flatten the class map to get the corresponding pixel values (labels)
    labels = class_map[indices[:, 0], indices[:, 1]]
    
    # Convert indices to a list of tuples for easier use
    indices = [tuple(idx) for idx in indices]
    
    return indices, np.array(labels.tolist()) - 1

################################################################################
def createImageCubes(X, indices, windowSize):
    """
    Extract patches centered at given indices from the hyperspectral image 
    after applying zero padding.
    
    Parameters:
    X (numpy array): Hyperspectral image of shape (N, M, P)
    indices (list of tuples): List of indices where patches should be extracted
    windowSize (int): Window size, the patch will be of size (windowSize, windowSize)
    
    Returns:
    list: List of image patches extracted from the padded hyperspectral image
    """
    # Calculate margin based on window size
    margin = windowSize // 2
    
    # Apply zero padding to the hyperspectral image
    N, M, P = X.shape
    X_padded = np.zeros((N + 2 * margin, M + 2 * margin, P))
    
    # Offsets to place the original image in the center of the padded image
    x_offset = margin
    y_offset = margin
    X_padded[x_offset:N + x_offset, y_offset:M + y_offset, :] = X
    
    # Extract patches centered at the provided indices
    patches = []
    
    for idx in indices:
        i, j = idx
        i = i + margin
        j = j + margin
        # Get patch boundaries, ensuring the patch is centered at (i, j)
        i_min = i - margin  # Centered on the index, accounting for padding
        i_max = i_min + windowSize
        j_min = j - margin
        j_max = j_min + windowSize
        
        # Extract the patch
        patch = X_padded[i_min:i_max, j_min:j_max, :]
        

        patches.append(patch)
    
    return np.array(patches)

###############################################################################

def createImageCubes_cmplx(X, indices, windowSize):
    """
    Complex-Valued version of the original one
    """
    # Calculate margin based on window size
    margin = windowSize // 2
    
    # Apply zero padding to the hyperspectral image
    N, M, P = X.shape
    X_padded = np.zeros((N + 2 * margin, M + 2 * margin, P), dtype = 'complex64')
    
    # Offsets to place the original image in the center of the padded image
    x_offset = margin
    y_offset = margin
    X_padded[x_offset:N + x_offset, y_offset:M + y_offset, :] = X
    
    # Extract patches centered at the provided indices
    patches = []
    
    for idx in indices:
        i, j = idx
        i = i + margin
        j = j + margin
        # Get patch boundaries, ensuring the patch is centered at (i, j)
        i_min = i - margin  # Centered on the index, accounting for padding
        i_max = i_min + windowSize
        j_min = j - margin
        j_max = j_min + windowSize
        
        # Extract the patch
        patch = X_padded[i_min:i_max, j_min:j_max, :]
        

        patches.append(patch)
    
    return np.array(patches)
###############################################################################
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]),dtype=('complex64'))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def target(name):
    if name == 'FL' or name == 'FL':
        target_names = ['Water', 'Forest', 'Lucerne', 'Grass', 'Rapeseed',
                        'Beet', 'Potatoes', 'Peas', 'Stem Beans', 'Bare Soil', 'Wheat', 'Wheat 2', 
                        'Wheat 3', 'Barley', 'Buildings']
    elif name == 'SF':
        target_names = [ 'Bare Soil', 'Mountain', 'Water', 'Urban', 'Vegetation']
        
    return target_names 
    
def num_classes(dataset):
    if dataset == 'FL':
        output_units = 15
    elif dataset == 'SF':
        output_units = 5
    return output_units




def Patch(data,height_index,width_index, PATCH_SIZE):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    
    return patch

def getTrainTestSplit(X, y, pxls_num):
    X = np.array(X)
    if type(pxls_num) != list:
        pxls_num = [pxls_num]*len(np.unique(y))
        
    if len(np.unique(y)) != len(pxls_num):
        print("length of pixels list doen't match the number of classes in the dataset")
        return
    else:
        xTrain = []
        yTrain = []
    
        xTest  = []
        yTest  = []
        for i in range(len(np.unique(y))):
            print(i)
            if pxls_num[i] > len(y[y==i]):
                print("Number of training pixles is larger than class pixels")
            #    return
            else:
                random.seed(321) #optional to reproduce the data
                samples = random.sample(range(len(y[y==i])), pxls_num[i])
                xTrain.extend(X[y==i][samples])
                
                yTrain.extend(y[y==i][samples])
        
             
                tmp1 = list(X[y==i])
               
                tmp3 = list(y[y==i])
                for ele in sorted(samples, reverse = True):
                    del tmp1[ele]
                    del tmp3[ele]

                xTest.extend(tmp1)
                yTest.extend(tmp3)

                
    xTrain, yTrain = shuffle(xTrain, yTrain, random_state=321)  
    xTest,  yTest = shuffle(X, y, random_state=345)
    
       
      
    return xTrain, yTrain, xTest, yTest
        
        
    
import cvnn.layers as complex_layers
def cmplx_SE_Block(xin, se_ratio = 8):
    # Squeeze Path
    xin_gap =  GlobalCmplxAveragePooling2D(xin)
    sqz = complex_layers.ComplexDense(xin.shape[-1]//se_ratio, activation='cart_relu')(xin_gap)
    
    # Excitation Path
    excite1 = complex_layers.ComplexDense(xin.shape[-1], activation='cart_sigmoid')(sqz)
    
    out = tf.keras.layers.multiply([xin, excite1])
    
    return out
    
   

import tensorflow as tf
def GlobalCmplxAveragePooling2D(inputs):
    inputs_r = tf.math.real(inputs)
    inputs_i = tf.math.imag(inputs)
    
    output_r = tf.keras.layers.GlobalAveragePooling2D()(inputs_r)
    output_i = tf.keras.layers.GlobalAveragePooling2D()(inputs_i)
    
    if inputs.dtype == 'complex' or inputs.dtype == 'complex64':
           output = tf.complex(output_r, output_i)
    else:
           output = output_r
    
    return output




def Standardize_data(X):
    new_X = np.zeros(X.shape, dtype=(X.dtype))
    _,_,c = X.shape
    for i in range(c):
        new_X[:,:,i] = (X[:,:,i] - np.mean(X[:,:,i])) / np.std(X[:,:,i])
        
    return new_X
        
        



from numpy.fft import fft2, fftshift
def getFFT(X):
    X_fft = np.zeros(X.shape, dtype='complex64')
    for ii in range(len(X)):
        for jj in range(X.shape[3]):
            X_fft[ii,:,:,jj] = fftshift(fft2(X[ii,:,:,jj])) 
            #X_fft[ii,:,:,jj] = fftshift(fft2(X[ii,:,:,jj])) 
            
            
    return X_fft


import keras
def cart_gelu(x):
    x_r = tf.math.real(x)
    x_i = tf.math.imag(x)
    
    gelu_r = keras.activations.gelu(x_r, approximate=False)
    gelu_i = keras.activations.gelu(x_i, approximate=False)
    
    if x.dtype == 'complex' or x.dtype == 'complex64':
           output = tf.complex(gelu_r, gelu_i)
    else:
           output = gelu_r
    
    return output


def predict_by_batching(model, input_tensor, batch_size):
    '''
    Function to to perform predictions by dividing large tensor into small ones 
    to reduce load on GPU
    
    Parameters
    ----------
    model: The model itself with pre-trained weights.
    input_tensor: Tensor of diemnsion batches x windowSize x windowSize x channels x 1.
    batch_size: integer value smaller than batches .

    Returns
    -------
    Predicetd labels
    '''
    
    num_samples = input_tensor.shape[0]
    k = 0
    predictions = []
    for i in range(0, num_samples, batch_size):
        print("batch", k, " out of", num_samples//batch_size)
        print(k*batch_size, "out of", num_samples )
        k+=1
        batch = input_tensor[i:i + batch_size]
        batch_predictions = model.predict(batch, verbose=1)
        predictions.append(batch_predictions)
        
    Y_pred_test = np.concatenate(predictions, axis=0)
  
    return Y_pred_test




import tensorflow as tf

def extract_polarimetric_features(T):
    """
    Extracts 12 polarimetric feature descriptors from the 6-channel PolSAR coherence matrix.
    
    Parameters:
    - T: A TensorFlow tensor of shape (batch_size, H, W, 6), 
         where the last dimension represents complex channel.
    
    Returns:
    - A tensor of shape (batch_size, H, W, 12) containing the computed real-valued descriptors.
    """
    # Convert real-imaginary representation into complex numbers
    

    # Compute the amplitude (magnitude) of each complex channel (First 6 Features)
    RF1 = tf.abs(T[..., 0])  # |T11|
    RF2 = tf.abs(T[..., 1])  # |T22|
    RF3 = tf.abs(T[..., 2])  # |T33|
    RF4 = tf.abs(T[..., 3])  # |T12|
    RF5 = tf.abs(T[..., 4])  # |T13|
    RF6 = tf.abs(T[..., 5])  # |T23|

    # Compute SPAN (Total Power)
    span = RF1 + RF2 + RF3  # SPAN = |T11| + |T22| + |T33|

    # RF7: Logarithmic SPAN
    RF7 = 10 * tf.math.log(span + 1e-6) / tf.math.log(10.0)

    # RF8, RF9: Normalized Ratios
    RF8 = RF2 / (span + 1e-6)  # T22 / SPAN
    RF9 = RF3 / (span + 1e-6)  # T33 / SPAN

    # RF10: T12 Relative Correlation Coefficient
    RF10 = RF4 / tf.sqrt((RF1 + 1e-6) * (RF2 + 1e-6))

    # RF11: T13 Relative Correlation Coefficient
    RF11 = RF5 / tf.sqrt((RF1 + 1e-6) * (RF3 + 1e-6))

    # RF12: T23 Relative Correlation Coefficient
    RF12 = RF6 / tf.sqrt((RF2 + 1e-6) * (RF3 + 1e-6))
    
    RF = tf.stack([RF1, RF2, RF3, RF4, RF5, RF6, RF7, RF8, RF9, RF10, RF11, RF12], axis=-1)
    #RF = Standardize_data(RF)
    # Stack the computed features into a single tensor
    return RF



from collections import defaultdict

def split_train_test(indexes, labels, pxls_num):
    # Ensure inputs are in correct format
    indexes = np.array(indexes)
    labels = np.array(labels)

    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)

    train_idx_all, test_idx_all = [], []

    for cls, cls_indices in class_to_indices.items():
        cls_indices = np.array(cls_indices)
        np.random.shuffle(cls_indices)

        train_idx = cls_indices[:pxls_num]
        test_idx = cls_indices[pxls_num:]

        train_idx_all.extend(train_idx)
        test_idx_all.extend(test_idx)

    # Select indexes and labels
    xTrain = [tuple(indexes[i]) for i in train_idx_all]
    yTrain = labels[train_idx_all]
    xTest = [tuple(indexes[i]) for i in test_idx_all]
    yTest = labels[test_idx_all]

    xTrain, yTrain = shuffle(xTrain, yTrain, random_state=321)  
    xTest,  yTest = shuffle(xTest, yTest, random_state=345)
    
    return xTrain, yTrain, xTest, yTest









