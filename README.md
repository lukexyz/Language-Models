# NLP Language Models
Sentiment analysis and generation using ULMFiT (2018) universal language models

# Methodology
The ULMFiT NLP transfer learning technique, was introduced in this 2018 paper (https://arxiv.org/pdf/1801.06146.pdf)


## Installation on AWS
`Deep Learning AMI (Ubuntu 16.04) Version 25.3`, GPU `p2.xlarge` for training :ballot_box_with_check:, `120 GB`

##### SSH into new linux box, create conda environment
    $ ssh -i "<key>.pem" ubuntu@ec2-<public-ip>.us-east-2.compute.amazonaws.com
    $ conda create -n fastai python=3.7
    $ conda activate fastai

##### Dependencies
    $ conda install jupyter notebook -y
    $ conda install -c conda-forge jupyter_contrib_nbextensions
    $ conda install fastai pytorch=1.0.0 -c fastai -c pytorch -c conda-forge

##### Update jupyter kernels (optional)
    $ conda install nb_conda_kernels -y
    $ python -m ipykernel install --user --name fastai --display-name "fastai v1"
    $ conda install ipywidgets

##### Validate installation
    $ python -m fastai.utils.show_install

##### Run notebooks
    $ jupyter notebook --ip=0.0.0.0 --no-browser
    
    # http://<public IP>:8888/?token=<token>


## Improvements
The 2019 paper [`MultiFiT: Efficient Multi-lingual Language Model Fine-tuning`](https://arxiv.org/abs/1909.04761) expanded on the `ULMFiT` method using  
1. `Subword Tokenization`, which uses a mixture of character, subword and word tokens, depending on how common they are. These properties allow it to fit much better to multilingual models (non-english languages).
    
<p align="center">
  <img src="https://github.com/lukexyz/Language-Models/blob/master/img/multifit_vocabularies.png?raw=true" width="300">
</p>

2. Updates the `AWD-LSTM` base RNN network with a `Quasi-Recurrent Neural Network` (QRNN). The QRNN benefits from attributes from both a CNN and an LSTM:
* It can be parallelized across time and minibatch dimensions like a CNN (for performance boost) 
* It retains the LSTM’s sequential bias (the output depends on the order of elements in the sequence).  
    `"In our experiments, we obtain a 2-3x speed-up during training using QRNNs"`

<p align="center" >
  <img src="https://github.com/lukexyz/Language-Models/blob/master/img/multifit_qrnn.png?raw=true" width="500">
</p>

Reference: `Efficient multi-lingual language model fine-tuning` 10 Sep 2019 by Sebastian Ruder and Julian Eisenschlos (http://nlp.fast.ai/classification/2019/09/10/multifit.html) 


