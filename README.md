# adversarial-cnn-cryptography
## A project for Physics 555 <br>

### Repository Contents <br>
project.py               -- main project code <br>
plots.py                 -- result processing and plotting <br>
/Images                  -- directory containing plots of results <br>
/errorFiles              -- directory containing numpy arrays of results for different total epoch values <br>
projectPresentation.pptx -- project presentation powerpoint <br>

### Project Summary
The aim of this course project is to reproduce some results from a paper entitled "Learning to Protect Communications with Adversarial Neural Cryptography" (Abadi & Andersen, 2016). The objective of this work is to ascertain whether neural networks can learn to communicate securely using secret keys. In other words, the question is: if neural networks A and B are given a secret key, can they use that key to send information from A to B without an intercepting neural network (E) being able to decipher the message. 

There are multiple applications to this type adversarial neural cryptography. For example, for reasons of privacy it may be necessary to prevent a component of a neural network from being able to access some aspect of its input data. In this scenario, the neural network component is the adversary and if the postulated question is true, the input data may be encrypted and later decrypted such that the component is not able to interpret the data.

Results were measured by comparing the reconstruction error evolution for the two receiving networks (B & E) against that detailed in Abadi & Andersen. Plots of my reconstruction errors can be found in the Images directory and their comparison against Abadi & Andersen is shown in projectPresentation.pptx

### Reproducibility Requirements: <br>
Python 3 <br>
Tensorflow 1.14 (eager execution disabled)<br>
numpy<br>
matplotlib <br>

### Resources I referenced: <br>
Abadi & Andersen 2016    -- https://arxiv.org/pdf/1610.06918.pdf <br>
Custom Training Tutorial -- https://www.tensorflow.org/tutorials/customization/custom_training <br>
Tensorflow Tutorial      -- https://adventuresinmachinelearning.com/python-tensorflow-tutorial/ <br>
jkbestami CryptoNN       -- https://github.com/jkbestami/CryptoNN

