
# ``LSTM-CNN-SUD v1.0``
## Hybrid biLSTM and CNN architecture for Sentence Unit Detection


Example code and data for paper [A Hybrid Deep Learning Architecture for Sentence Unit Detection](https://ieeexplore.ieee.org/abstract/document/8629178) (IALP 2018).

LSTM-CNN-SUD, version ``1.0``, is a project that was developed with 3 main functions:

- Detect sentence unit in an unpunctuated sequence of text
- Train new models with given corpora that follow the format
- Evaluate trained models with test dataset

We evaluated our model on 2 datasets: **RT-03-04** (transcripts and annotations of 40 of hours English Broadcast News and Conversational Telephone Speech audio data) and **MGB** (1,340 hours over 1,600 hours of broadcast audio text transcriptions taken from four BBC TV channels on seven weeks). Because these datasets are closed, we use example data for this project.


## Table of Contents
- [LSTM-CNN-SUD v1.0](#lstm-cnn-sud-v10)
  * [1. Installation](#1-installation)
  * [2. Usage](#2-usage)
    + [Train the models](#train-the-models)
    + [Evaluate the trained models](#evaluate-the-trained-models)
##

## 1. Installation

This program was developed using **Python** version **3.5** and was tested on **Ubuntu 16.04** system. We recommend using Anaconda 3 newest version for installing **Python 3.5** as well as **numpy**, although you can install them by other means. 

Other requirements: 
 1. **numpy**
```sh
# Included in Anaconda package
```

 2. **scipy**
```sh
# Included in Anaconda package
```

 3. **h5py** 
```sh
$ conda install -c anaconda h5py 
```

 4. **Tensorflow** 
```sh
$ pip install tensorflow     # Python 3.n; CPU support 
$ pip install tensorflow-gpu # Python 3.n; GPU support 
```
If you are install tensorflow with GPU support, please follow the instructions on the official document to install other required libraries for your platform. Official document can be found at [https://www.tensorflow.org/install/](https://www.tensorflow.org/install/)  

 5. **Keras**
```sh
$ pip install keras 
```

 6. **sklearn**
```sh
$ conda install -c anaconda scikit-learn
```

## 2. Usage 
### Train the models
Use testing file ``test.py`` to train the model and evaluate on benchmark dataset.
Commands: 
```sh
$ python test.py --help
usage: test.py [-h] [-i I] [-e E] [-p P] [-b B] [-ft FT] [-cnn CNN]
                      [-hd HD]

Hybrid biLSTM and CNN architecture for Sentence Unit Detection

optional arguments:
  -h, --help  show this help message and exit
  -i I        Job identity
  -e E        Number of epochs
  -p P        Patience of early stop (0 for ignore early stop)
  -b B        Batch size
  -ft FT      Number of output fastText w2v embedding LSTM dimension
  -cnn CNN    CNN configurations
  -hd HD      Hidden layer configurations
```

All hyper-parameter is set default to the tuned values. You can change every setting of the model or try the default one. 

**Example 1**: test with default setting
```sh
$ python test.py
```

**Example 2**: test with some configurations
```sh
$ python test.py –lstm 256,128 –cnn 1:32,3:64,5:32 –hd 128,64
```
This command means:
 - Config the LSTM using 2 bidirectional LSTM layers with dimensions of hidden state in each layer are 256 and 128 corresponding.
 - Use 3 different region sizes (1, 3, 5) of CNN with 32, 64 and 32 filters corresponding.
 - Using 2 hidden layers before softmax layer with 128 and 64 nodes.

### Evaluate the trained models 
After train the model, the model is used to predict on the provided testing dataset. The result is printed out in the end of the output stream.

The result is printed in the format: 
```sh
...
[INFO] Training model... finished in x s.
Testing model over test set
Result:	P=x%	R=x%	F1=x%
```
