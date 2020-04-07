# -*- coding: utf-8 -*-
"""
Created on Thursday March 5 22:09:59 2020

@author: Alexandra Borukhovetskaya

See attached README.txt file for project details
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense,Conv1D,Lambda

tf.reset_default_graph()

# Set parameters as given in paper
LEARNING_RATE   = 0.0008
BATCH_SIZE      = 4096
SAMPLE_SIZE     = 4096
EPOCHS          = 200000 #should be set to 850000, resources permitting
STEPS = int(SAMPLE_SIZE/BATCH_SIZE)

TEXT_SIZE = 16
KEY_SIZE  = 16

def genBools(size, n):
    """
    Draw boolean values from a discrete uniform distribution

    Parameters
    ----------
    size : int
        Output shape; number of arrays returned
    n : int
        Output shape; length of each array

    Returns
    -------
    ndarray of np.float32
        A sample of messages/keys composed of values that are either 1 or -1.

    """
    randBools = np.random.randint(0,2,size=[size,n])
    
    #convert each 0 to -1 (because 0 indicates signal absence)
    randBools = randBools*2-1
    return randBools.astype(np.float32)

def getBools(sSize, mSize, kSize):
    """
    Implement the genBools function to create samples of messages and keys

    Parameters
    ----------
    sSize : int
        Sample size.
    mSize : int
        Message size.
    kSize : int
        Key size.

    Returns
    -------
    m : ndarray of np.float32
        A sample of plaintext messages to be fed into Alice.
    k : ndarray of np.float32
        A sample of cipher keys to be fed into Alice and Bob.

    """
    m = genBools(sSize, mSize)
    k = genBools(sSize, kSize)
    return m,k

def model(collection, message, key=None):
    """
    Generate a model for Alice, Bob, or Eve

    Parameters
    ----------
    collection : string
        either 'Alice', 'Bob', or 'Eve'.
    message : 'tensorflow.python.framework.ops.Tensor'
        tf.float32, shape=(BATCH_SIZE, TEXT_SIZE), input plaintext message for Alice.
    key : 'tensorflow.python.framework.ops.Tensor', optional
        tf.float32, shape=(BATCH_SIZE, TEXT_SIZE), cipher key for Alice and Bob. The default is None.

    Returns
    -------
    'tensorflow.python.framework.ops.Tensor'
        model output given specified input.

    """
    if(key is not None): #Alice or Bob
        modelInput = tf.concat(axis=1, values=[message, key])
        shp = 32
    else: #Eve
        modelInput = message
        shp = 16
    
    with tf.variable_scope(collection):
        model = tf.keras.models.Sequential([
          #fully-connected layer of 32 neurons
          Dense(TEXT_SIZE+KEY_SIZE,activation='relu',input_shape=(shp,)),
          #expand layer
          Lambda(lambda fc: tf.expand_dims(fc, axis=2)),
          # 3 1D concolutional layers with sigmoid activation
          Conv1D(filters=2,kernel_size=4,padding='same',activation='sigmoid'),
          Conv1D(filters=4,kernel_size=2,strides=2,activation='sigmoid'),
          Conv1D(filters=4,kernel_size=1,padding='same',activation='sigmoid'),
          # 1 1D convolutional layer with tanh activation to produce "binary" output
          Conv1D(filters=1,kernel_size=1,padding='same',activation='tanh'),
          #reverse above expand layer
          Lambda(lambda c4: tf.squeeze(c4,axis=2))
      ])
    return model(modelInput)

def abLoss(aIn,bOut,eLoss):
    """
    Loss function for Alice and Bob

    Parameters
    ----------
    aIn : 'tensorflow.python.framework.ops.Tensor'
        plaintext message; input of Alice.
    bOut : 'tensorflow.python.framework.ops.Tensor'
        plaintext message; output of Bob.
    eLoss : 'numpy.float32'
        Eve's loss as given by the function eLoss().

    Returns
    -------
    loss : 'numpy.float32'
        Alice and Bob's loss.

    """
    diff = (bOut+1.)/2.-(aIn+1.)/2.
    bLoss = (1./BATCH_SIZE)*tf.reduce_sum(tf.abs(diff))
    eavesLoss = tf.reduce_sum(tf.square(float(TEXT_SIZE)/2.-eLoss)/(TEXT_SIZE/2.)**2.)
    loss = bLoss + eavesLoss
    return loss

def eLoss(aIn,eOut):
    """
    Loss function for Eve

    Parameters
    ----------
    aIn : 'tensorflow.python.framework.ops.Tensor'
        plaintext message; input of Alice.
    eOut : 'tensorflow.python.framework.ops.Tensor'
        plaintext message; output of Eve.

    Returns
    -------
    loss : 'numpy.float32'
        Eve's loss.

    """
    diff = (eOut+1.)/2.-(aIn+1.)/2.
    loss = (1./BATCH_SIZE)*tf.reduce_sum(tf.abs(diff))
    return loss

#placeholder that will be updated in sess.run using feed_dict contents
aIn = tf.placeholder(tf.float32, shape=(BATCH_SIZE, TEXT_SIZE), name='plaintextMessage')
key = tf.placeholder(tf.float32, shape=(BATCH_SIZE, KEY_SIZE), name='cipherKey')

#generate our scenario
aOut = model('Alice', aIn, key)
bOut  = model('Bob', aOut, key)
eOut  = model('Eve', aOut)

## Compute Eve's loss
eveLoss = eLoss(aIn,eOut)

## Compute Alice and Bob's loss
bobLoss = abLoss(aIn,bOut,eveLoss)

# Collect trainable tensors
aVars =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Alice') 
bVars   =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Bob') 
eVars   =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Eve') 

# Define Adam optimizer using parameters given in the paper
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, epsilon=1e-08)

#training
trainEve  = optimizer.minimize(eveLoss, var_list=[eVars])
trainBob  = optimizer.minimize(bobLoss, var_list=[aVars + bVars])

# begin tensorflow session used to run process given above
sess = tf.Session() 
init = tf.global_variables_initializer()
sess.run(init)

# Generate input data
messages,keys = getBools(SAMPLE_SIZE, TEXT_SIZE, KEY_SIZE)

# Define arrays to record losses for analysis and plotting
bobWrong = np.zeros(STEPS*EPOCHS)
eveWrong = np.zeros(STEPS*EPOCHS)

# Begin custom training loop
for i in range(EPOCHS):
  for j in range(STEPS):

    # define batch data
    batchMessages = messages[j*BATCH_SIZE: (j+1)*BATCH_SIZE]
    batchKeys     = keys[j*BATCH_SIZE: (j+1)*BATCH_SIZE]

    # Alice and Bob Training
    runAB = sess.run([trainBob, bobLoss],feed_dict={aIn:batchMessages, key:batchKeys })
    abLossCurrent = runAB[1]

    # Eve Training
    for k in range(2): # train Eve twice for each training of Alice & Bob
      runE = sess.run([trainEve, eveLoss], feed_dict={aIn:batchMessages, key:batchKeys })
      eLossCurrent = runE[1]

    # update loss history
    bobWrong[i*STEPS+j] = abLossCurrent
    eveWrong[i*STEPS+j] = eLossCurrent
    
  # display bit error every 50 epochs
  if(i%50==0):
    print('   epochs: ',i, '   bob bits wrong: ',abLossCurrent,
          '   eve bits wrong:',eLossCurrent)

sess.close()

#save loss history for plotting in plots.py
np.save('bobWrong_'+str(EPOCHS),bobWrong)
np.save('eveWrong_'+str(EPOCHS),eveWrong)
