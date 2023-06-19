# Pytorch-pretrained-models
We wrap various pretrained models in pytorch with lightning and neptune logger (developed by AISeed ) 

## prequisite
- Python=3.9
- CUDA: 11.2/11.3
- Pytorch framwork: 1.12.1, pytorch-lightning
- Others: numpy, opencv, scipy
- dashboard: neptune ai (for training), gradio (for testing)
## Environments Settings
- Install [Anaconda](https://www.anaconda.com/)
- create a new environment:
```
conda create --name=ENV_NAME python=3.9
conda activate ENV_NAME
```
- Install dependencies: 
```
conda env create -f environment.yml
```
#

It facilitates three distinct modes of operation: (1) inference utilizing the complete model, (2) fine-tuning with a particular class within the standard layers. Table 1 presents comprehensive information regarding the supported models, while the subsequent section outlines the configurations to be provided.
- `Classifier.py`: [WrappingClassifier](Modules/Classifier.py#L59) Wrapping Head for classification task.
- `Detector.py`: [WrappingDetector](Modules/Detector.py#L30) Wrapping Head for object detection task.
- `Segment.py`: [WrappingSegment](Modules/Segment.py#L22) Wrapping Head for semantic segmentation task.

![alt text](image/test.png)

<table>
<tr>
<td colspan=1>
    Table 1. Supported models
</td>

| Name  | Description |Metrics | Params |
| ------------- | ------------- | ------------- | ------------- |
| `Classification` | None| Accuracy | milions |
| resnet | [ResNet](https://arxiv.org/abs/1512.03385)  | 93.33  | s: 11.5, m: 28.0, l: 62.6 |
| efficientnet | [Efficient Net v2](https://arxiv.org/abs/2104.00298)  | 95.46  | s: 21.9, m: 54.6, l: 118|
| vgg | [Very Deep CNN](https://arxiv.org/abs/1409.1556)  | -  | s: 9.5, m: 9.7, l: 20.3 |
| densenet | [DenseNet](https://arxiv.org/abs/1608.06993)  |  - | s: 8.1, m: 31.7, l: 22.0|
| wide_resnet | [Wide ResNet](https://arxiv.org/abs/1605.07146)  | -  | s: 71.3, m: 129|
| inception | [Inception v3](https://arxiv.org/abs/1512.00567)  |  - | 29.6 |
| mobilenet | [Mobile Net v3](https://arxiv.org/abs/1905.02244)  | -  | s: 1.3, m:, l: 4.0|
| shufflenet | [Shuffle Net](https://arxiv.org/abs/1807.11164)  |  - | s: 1.5, m: 2.4, l: 9.8 |
| convnext | [ConvNext](https://arxiv.org/abs/2201.03545)  | -  | s: 28.5, m: 88.7, l: 198 |
| resnext | [ResNext](https://arxiv.org/abs/1611.05431v2)  |  - | s: 27.5, m: 91.2, l: 85.9 |
| vit | [Vision Transformer](https://arxiv.org/abs/2010.11929)  |  - | s: 86.4, m: 304, l: |
| swin | [Swin Transformer](https://arxiv.org/abs/2103.14030)  |  93.17 | s: 28.2, m: 49.5, l: 87.9|
| `Detection` | None| mAP | milions |
| retinanet | [Retina Net](https://arxiv.org/abs/1708.02002)  |  - | m: 32.2, l: 36.4 |
| ssd | [Single Shot Detection](https://arxiv.org/abs/1512.02325)  |  66.43 | s: 3.8, m: 25.4 |
| fasterrcnn |[Faster Region Proposlal CNN](https://arxiv.org/abs/1506.01497)  | 81.16  | s: 20.0, m: 42.4 |
| `Segmentation` | None| dice | milions |
| fcn | [Fully CNN](https://arxiv.org/abs/1411.4038)  |  79.81 | m: 28.0, l: 47.0 |
| deeplab | [Atrous Convolution](https://arxiv.org/abs/1706.05587) | 81.08  | m: 23.5, l: 87.9 |
| maskrcnn | [Masked RCNN](https://arxiv.org/abs/1703.06870) |  | m: 39.5, l:  |
#
## Training Interface
Examples of training Wrapping Network can be found in [ults.py](Modules/ultis.py) and Notebook [examples](examples.ipynb), we config hyper-parameters in [config.yaml](Modules/config.yaml) file

- `ultis.py`: Three pre-defined datasets have been established, each serving as a demonstration for the training-testing process of a specific task, [CIFAR10](Modules/ultis.py#) for classification, [Lung CT-scan](Modules/ultis.py) for object detection, and [PennFudan](Modules/ultis.py) for binary object segmentation.
- Notebook `examples`: Our main module is [ModulesModel](examples.ipynb) that based on [pytorch-lightning](https://lightning.ai/pages/open-source/) and logged by [neptune-ai](https://neptune.ai/). As shown in the Figure above, we logged hyperparameters, metrics, and results from each run.

![alt text](image/neptune.jpg)

```
Modify in Examples Notebook
```
#

## Testing Interface
We deploy (demo) our model using [Gradio](https://gradio.app/), which supports to visualize results from 3 tasks: classification, detection, and segmentation, depending on the selected model.

![alt text](image/gradio.png)

```
Modify in Example Notebook
```

#
## Configuration (Config file)
The configurations, a [config.yaml](Modules/config.yaml), encompassing the model architecture and training settings, as well as dataset settings. The "config.yaml" file follows a structured format, consisting of a list of dictionaries. Each dictionary within the list represents a distinct configuration and saves specific configuration parameters.

<table>
<tr>
<td colspan=1>
    Table 2. Configuration
</td>

| Parameters  | Description |Scope | Value |
| ------------- | ------------- | ------------- | ------------- |
| `Model` |
| name | Model's name  | architect_settings  | string |
| name | Pretrained model  | architect_settings/backbone  | string: "name"-"s/m/l" |
| is_full | If True, use full model  | architect_settings/backbone  | Bool |
| is_pretrained |  pretrained weights  | architect_settings/backbone  | Bool |
| is_freeze | Freeze weights  | architect_settings/backbone  | Bool |
| n_cls | num classes  | architect_settings | int |
| `Dataset` |
| name | Dataset name  | dataset_settings | string: "LungCT-Scan", "CIFAR10", "PennFudan" |
| path | path to dataset  | dataset_settings  | string |
| img_size | size of image to model  | dataset_settings  | int |
| `Training` |
| gpu_ids | list of gpus used  | training_settings  | list: [0] |
| n_gpu | num gpus  | training_settings  | int |
| img_size | size of image to model  | training_settings  | int |
| loss | loss function  | training_settings  | str: "ce" (classification/segmentation), "spread", "dice", "mse", "none"(detection) |
| ckpt_path | path to check-points  | training_settings  | str |
| n_epoch | num epoch  | training_settings  | int |
| n_batch | batch size  | training_settings  | int |
| num_workers | num workers to dataloader | training_settings  | int |
| optimizer | optimizer | training_settings  | str: "adam", "sgd" |
| lr_scheduler | learning rate scheduler | training_settings  | str: "step", "multistep", "reduce_on_plateau" |
| lr | learning rate | training_settings  | float|
| lr_step | learning rate step for decay| training_settings  | int|
| lr_decay | learning rate decay rate | training_settings  | float|
| momentum | momentum for optimizer | training_settings  | float|
| weight_decay | weight decay for "sgd" | training_settings  | float|
</tr>
The understanding of the functioning of the configuration file is best obtained by referring to the actual "config.yaml" file. It is crucial to acknowledge that the module's flexibility is maintained by avoiding excessive hard coding of parameters. This is because fine-tuning and optimizing the parameters play a vital role in the development process. As part of our ongoing efforts to enhance the performance of the module, we anticipate making further refinements to the configurations. Consequently, it is likely that the config file will be subject to future updates to reflect these optimizations.

#
# References
