# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 00:02:28 2020

@author: asya
"""

import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE      = 4096
SAMPLE_SIZE     = 4096
EPOCHS          = 85000
STEPS = int(SAMPLE_SIZE/BATCH_SIZE)

bW = np.load('bobWrong_85000.npy')
eW = np.load('eveWrong_85000.npy')

steps = np.arange(0,EPOCHS*STEPS,1)

plt.figure(figsize=(8,3))
plt.plot(steps,bW,label='Bob',c='red')
plt.plot(steps,eW,label='Eve',c='green')
plt.xlim(0,85000)
plt.xlabel('step')
plt.ylabel('bits wrong')
plt.title('reconstruction error evolution')
plt.legend()
plt.savefig('errorEvolution_xlim85000',dpi=140)