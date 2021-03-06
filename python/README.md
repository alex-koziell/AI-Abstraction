# Deep Learning "from scratch" in Python

Closely following https://course.fast.ai/part2, the aim is to develop a deep learning library built on Pytorch using jupyter notebooks, gathering some insights into deep learning along the way, including paper implementaions, performance analysis, and explainability.

The notebooks are being numbered to reflect stages in which the library is built. I plan to refactor code by making a new notebooks, to retain a nice record of the development process.

## Log
*__28/06/2020__ - Wrote a script to export code from jupyter notebooks into a python modules in a systematic way.*

*__29/06/2020__ - Created some small testing functions and familiarized self with fastai.datasets module.*

*__30/06/2020__ - Implemented forward and backward pass for a DNN using inspiration from Pytorch's `nn.Module`.*


*__01/07/2020__*
- *Data API for drawing either sequentially or randomly sampled batches from data (re-create Pytorch's `DataLoader`).*
- *New activations and loss functions (softmax, neg log likelihood, cross entropy).*
- *SGD optimizer*.
- *Model definition.*
- *Training/Eval loop.*
- *Neat and concise data loading, model definition and training/eval (just 4 lines of code!).*

*__03/07/2020__*
- *Callback API (a lot of work!)*
- *Running average stats and early stopping callbacks*

*__04/07/2020__*
- *Refactor callbacks and added cancel batch/epoch/fit control flow, using exception handling.*
- *Recorder callback, schedulers and annealing using callbacks.*

*__05/07/2020__ - Lambda layers and CNN model wrapper.*

*__06/07/2020__* 
- *Cuda Callback, batch transformation callback.*
- *Awesome manual hook implementation/recording activations of each CNN ReLU.*

*__08/07/2020__* 
- *Hooks class.*
- *Kaiming initialization to fix our crazy activations from notebook 09.*

*__09/07/2020__* - *More stats visualisations with histograms.*

*__10/07/2020__* - *General ReLU class.*

*__13/07/2020__* - *Batch normalization implementation.*

|  Notebooks  |  Features  |  Working  |
|-------------|:----------:|:---------:|
| *00_exports*  |  • Turn code from jupyter noteboooks to python modules  |  ✓ |
| *01_testing* | • Tests based on some comparable (supplied as a function argument). | ✓ |
| *02_MNISTLoader* | • Loads MNIST with training and validation split. | ✓ |
| *03_DNN* | • Dense neural network capable of forward and backward pass. | ✓ |
| *04_DataAPI* | • Sequential and Random Batch sampling of data. | ✓ |
| *05_(...)* | • Cross-entropy loss, SGD optimizer and model training/eval. | ✓ |
| *06_Callbacks* | • Callback API, avg stats, early stopping.   | ✓ |
| *07_Annealing* | • Record losses and parameters during training, schedulers and annealing using callbacks.   | ✓ |
| *08_LambdaLayers* | • Lambda layers, CNN model wrapper, Cuda and Batch Xform callbacks.   | ✓ |
| *09_Hooks* | • Manual hooks, layer wise activation plotting.   | ✓ |
| *10_Xavier_Kaiming_Inits* | • Xavier, Kaiming initialization techniques and explanation.   | ✓ |
| *11_MoreStats* | • Layer activation histograms.   | ✓ |
| *12_GeneralReLU* | • General ReLU class. | ✓ |
| *13_Batchnorm* | • Batch normalization. | ✓ |
| *- 01_datasets* | • All the required functionality from fastai.datasets | ✗ |
