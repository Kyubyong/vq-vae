# VQ-VAE

This is a Tensorflow Implementation of VQ-VAE Speaker Conversion introduced in [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937). Although the training curves look fine, the samples generated during training were bad. Unfortunately, I have no time to dig more in this as I'm tied with my other projects. So I publish this project for those who are interested in the paper or its implementation. If you succeed in training based on this repo, please share the good news.

## Data
  * [vctk](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)


## Requirements

  * NumPy >= 1.11.1
  * TensorFlow >= 1.3
  * librosa
  * tqdm
  * matplotlib
  * scipy


## Training
  * STEP 0. Download [vctk](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)
  * STEP 1. Adjust hyper parameters in `hyperparams.py`. 
  * STEP 2. Run `python prepro.py`. 
  * STEP 3. Run `python train.py`.

