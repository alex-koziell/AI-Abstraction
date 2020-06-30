# Deep Learning "from scratch" in Python

Closely following https://course.fast.ai/part2, the aim is to develop a deep learning library built on Pytorch using jupyter notebooks.

The notebooks are being numbered to reflect stages in which the library is built. I plan to refactor code by making a new notebooks, to retain a nice record of the development process.

## Log
*__28/06/2020__ - Wrote a script to export code from jupyter notebooks into a python modules in a systematic way.*

*__29/06/2020__ - Created some small testing functions and familiarized self with fastai.datasets module.*

*__30/06/2020__ - Implemented forward and backward pass for a DNN using inspiration from Pytorch's `nn.Module`.*


|  Notebooks  |  Features  |  Working  |
|-------------|:----------:|:---------:|
| *00_exports*  |  • Turn code from jupyter noteboooks to python modules  |  ✓ |
| *01_testing* | • Tests based on some comparable (supplied as a function argument). | ✓ |
| *02_MNISTLoader* | • Loads MNIST with training and validation split. | ✓ |
| *03_DNN* | • Dense neural network capable of forward and backward pass. | ✓ |
| *- 01_datasets* | • All the required functionality from fastai.datasets | ✗ |
