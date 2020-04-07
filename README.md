# adversarial-cnn-cryptography
A project for Physics 555

The aim of this course project is to reproduce some results from a paper entitled "Learning to Protect Communications with Adversarial Neural Cryptography" (Abadi & Andersen, 2016). The objective of this work is to ascertain whether neural networks can learn to communicate securely using secret keys. In other words, the question is: if neural networks A and B are given a secret key, can they use that key to send information from A to B without an intercepting neural network being able to decipher the message. 

There are multiple applications to this type adversarial neural cryptography. For example, it may be necessary to prevent a component of a neural network from being able to access some aspect of its input data. In this scenario, the neural network component is the adversary and if the postulated question is true, the input data may be encrypted and later decrypted such that the component is not able to interpret the data.

Reproducibility Requirements: <br>
Python 3 <br>
Tensorflow 1.14 (eager execution disabled)<br>
numpy<br>
matplotlib <br>

Resources I referenced: <br>
Abadi & Andersen 2016 - https://arxiv.org/pdf/1610.06918.pdf <br>
Custom Training Tutorial - https://www.tensorflow.org/tutorials/customization/custom_training <br>
Tensorflow Tutorial - https://adventuresinmachinelearning.com/python-tensorflow-tutorial/ <br>
jkbestami CryptoNN - https://github.com/jkbestami/CryptoNN

