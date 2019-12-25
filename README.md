# Dogs_vs_Cats_Pytorch
# Introduction
- This repository is for kaggle `Dogs vs. Cats` match, but you can utilize this code to learn how to use `pytorch`. 
- For network, I has estabilished the structure containing the introduction of pre-trained models like `VGG`,`Inceptionv3` and `ResNet`.
- For optimizer, only `Adam` is illustrated in my repository.

# Environment
- Python 3.6
- Pytorch 0.4.0
- torchvision 0.2.0
- tensorboardX 1.5
    
# Dataset Structure
## Original
```
${project_dir}/datasets
    dogs_vs_cats
        train.zip
        test1.zip
```
## Extract train and test datasets
After downloading the datasets from Kaggle website, you need to extract these two zips.(Actually, I just extract train.zip)
```
${project_dir}/datasets
    dogs-vs-cats
        train.zip
        test1.zip
        Dataset           # Extracted from train.zip
        test1           # Extracted from test1.zip
```
## Final step
- In fact, the filenames of train and test datasets is in different naming conventions.
- For train one, the filename name is in cat.x.jpg or dog.x.jpg. However, x.jpg is used in test dataset. 
- To do the classification task easily, I just use the train dataset(Dataset). So if you want to build a new test/val dataset. You need to run `redistribution_image.py` to split the train dataset into a new one and an extra test dataset.
```
${project_dir}/datasets
    dogs-vs-cats
        train.zip
        test1.zip
        train           # Separated from 'train' directory by run `redistribution_image.py`.
        val             # Separated from 'train' directory by run `redistribution_image.py`.
        Dataset         # Extracted from train.zip
        test1           # Extracted from test1.zip
```
# Experimental Directory Structure
- Before training, you need to modify the directories in `demo.py`
- Run `demo.py` to start the training process. The follow directories will be created automatically.
```
${project_dir}
    checkpoint
        inceptionv3
            ckpt.t7
        resnet50
            ckpt.t7
        vgg
            ckpt.t7
    log(tensorboard dirctory)
        inceptionv3
        resnet50
        vgg
    model
        inceptionv3
            net_ 1.pth
            net_ 2.pth
            ...
        resnet50
            net_ 1.pth
            net_ 2.pth
            ...
        vgg
            net_ 1.pth
            net_ 2.pth
            ...
    record
        inceptionv3
            acc.txt
            log.txt
        resnet50
            acc.txt
            log.txt
        vgg
            acc.txt
            log.txt
            ...
```
# TensorboardX
- You can walk into `log\$(use_model)` directory to monitor the loss. Run `tensorboard --logdir .` then open the browser.
# Performances
| Network | Test Accuracy(%)| batch_size |
|---|---|---|
| VGG19 | 96.00 | 4 |
| ResNet50 | 96.00 | 32 |
| InceptionV3 | 76.00 | 32 |
- I has just trained the models for 1 epochs by 'Adam'.
# Attention
If the train or test accuracy is low, you can modify the optimizer code to 

`optimizer = torch.optim.Adam(use_model.parameters())` 

to let all the parameters for training.

Besides, you need to comment the following code:
```
for parma in use_model.feature.parameters():
    parma.requires_grad = False
for index, parma in enumerate(use_model.classifier.parameters()):
    if index == 6:
        parma.requires_grad = True
```
    


