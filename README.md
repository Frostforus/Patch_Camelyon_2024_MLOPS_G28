# Patch Camelyon 2024 MLOps group 28

## OVERALL GOAL OF THE PROJECT:
The goal of this project is to become familiar with skills associated with Machine Learning Projects, and during that create an acceptable classifier with the skills learned. This classifier will be a binary classifier based on the PatchCamelyion benchmark, the model will answer if a picture has a tumor or not.

## FRAMEWORK: 
We are planning on using pytorch extended with torchvision. As pytorch is the most used framework in academic settings, and it was recommended in this course, we decided to go with it for ease of use.  For image classification a natural choice with pytorch is torchvision, as it is built for the framework.
Torchvision: https://pytorch.org/vision/stable/index.html

### Tutorials for torchvision:
https://pytorch.org/vision/stable/auto_examples/index.html

## DATASET: https://github.com/basveeling/pcam

The PatchCamelyon benchmark is a new and challenging image classification dataset. It consists of 327.680 color images (96 x 96px) extracted from histopathologic scans of     lymph node sections. Each image is annoted with a binary label indicating presence of metastatic tissue.

## MODEL: 

Initially we will start with a decoder model based of the VGG architecture [link](https://pytorch.org/vision/stable/models/vgg.html)

### VGG:

The VGG (Visual Geometry Group) architecture is a deep convolutional neural network (CNN) renowned for its simplicity and effectiveness in image classification tasks. Developed by the Visual Geometry Group at the University of Oxford, VGG consists of several convolutional layers, followed by max-pooling layers for spatial downsampling. In the context of binary classification, the VGG architecture can be tailored by adjusting the output layer to produce binary predictions. Fine-tuning the pre-trained VGG model on a specific dataset can optimize its performance for the binary classification task. Leveraging the deep representation capabilities of VGG, it becomes a robust choice for discerning features and patterns in images, facilitating accurate binary classification task. [The Chad](https://chat.openai.com/share/ac106aed-2abc-4bb8-b7f5-e2a321d12054)



## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── patch_camelyon_2024_mlops_g28  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
