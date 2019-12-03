# Deep-Sense-NLP
Sentiment analysis and generation using ULMFiT (2018) universal language models

# Methodology


## Installation on AWS
`Deep Learning AMI (Ubuntu 16.04) Version 25.3`, GPU `p2.xlarge` for training :ballot_box_with_check:, `100 GB`

##### SSH into new linux box, activate pytorch conda environment
    $ ssh -i "<key>.pem" ubuntu@ec2-<public-ip>.us-east-2.compute.amazonaws.com

##### Create conda environment
    $ conda create -n fastai python=3.7
    $ conda activate fastai

##### Dependencies
    $ conda install jupyter notebook -y
    $ conda install -c conda-forge jupyter_contrib_nbextensions
    $ conda install fastai pytorch=1.0.0 -c fastai -c pytorch -c conda-forge

##### Update jupyter kernels (optional)
    $ conda install nb_conda_kernels
    $ python -m ipykernel install --user --name fastai --display-name "fastai v1"
    $ conda install ipywidgets

##### Validate installation
    $ python -m fastai.utils.show_install

## Run notebooks (with open port to home IP)
    $ jupyter notebook --ip=0.0.0.0 --no-browser
    
    # http://<public IP>:8888/?token=<token>
