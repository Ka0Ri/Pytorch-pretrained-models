# Pytorch-pretrained-models (AISeed)
[![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)](https://www.python.org/)
[![Pytorch 3.9](https://img.shields.io/badge/pytorch-1.12-orange.svg?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://www.pytorchlightning.ai/index.html)
[<img src="https://img.shields.io/badge/Dockerhub-blue.svg?logo=docker&logoColor=white">](https://www.docker.com/)
<!-- [<img src="https://img.shields.io/badge/LABEL-MESSAGE-COLOR.svg?logo=LOGO">](<LINK>) -->

The project aims to provide a framework for efficiently utilizing pre-trained models in PyTorch using the PyTorch Lightning library. Additionally, it incorporates the Neptune logger. This project serves two primary purposes: to facilitate rapid testing of new datasets and to act as a framework for downstream tasks by leveraging pre-trained models.

## Features
- Integration of pre-trained models into PyTorch Lightning
- Support for seamless experimentation with new datasets
- Logging and monitoring capabilities through Neptune logger
- Simplified framework for downstream tasks with pre-trained models

## Installation

### prequisite
- Python=3.9
- CUDA: 11.2/11.3
- Pytorch framwork: 1.12.1, pytorch-lightning
- Others: numpy, opencv, scipy
- dashboard: neptune ai (for training), gradio (for testing)
### Environments Settings
1. Install [Anaconda](https://www.anaconda.com/)
- create a new environment:
```
conda create --name=ENV_NAME python=3.9
conda activate ENV_NAME
```
2. Clone the repository:
```bash
git clone https://github.com/Ka0Ri/Pytorch-pretrained-models.git
```
3. Install dependencies: 
```bash
pip install -r requirements.txt
```
### Docker (Linux only)
1. [Install Docker](https://docs.docker.com/engine/install/ubuntu/)
- Download Docker
```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```
```bash
sudo usermod -aG docker $USER
```
- Install CUDA container runtime
```bash
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list |\
sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime
```
- Restart Docker
```bash
sudo systemctl stop docker
sudo systemctl start docker
```
2. Pull the Image
```bash
docker build --tag pytorch-finetune:runtime --file Dockerfile .
```
3. Check docker environment
```bash
docker run --rm -it --init \
  --gpus=all \
  pytorch-finetune:runtime nvidia-smi
```

## Usage

### Model

We implement 3 wrappping classes based on 3 tasks, classification, detection and segmentation.

- `Classifier.py`: [ClassificationModel](Modules/model/Classifier.py) Wrapping Head for classification task.
- `Detector.py`: [DetectionModel](Modules/model/Detector.py) Wrapping Head for object detection task.
- `Segment.py`: [SegmentModel](Modules/model/Segment.py) Wrapping Head for semantic segmentation task.

![alt text](assets/test.png)

Table 1 shows the supported models, [torcvision.models](https://pytorch.org/vision/0.8/models.html), while the subsequent section outlines the configurations to be provided.

<table>
<tr>
<td colspan=1>
    Table 1. Supported models.
</td>

| class  | models |weight | version |
| ------------- | ------------- | ------------- |------------- |
| `Classification` | | | |
| [models.ResNet](https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html) | resnet{x}|ResNet{x}_Weights.DEFAULT|`18` `34` `50` `34` `101` `152`|
| [models.DenseNet](https://pytorch.org/vision/0.8/_modules/torchvision/models/densenet.html) | densenet{x}|DenseNet{x}_Weights.DEFAULT|`121` `169` `161` `201` |
| [models.Inception3](https://pytorch.org/vision/0.8/_modules/torchvision/models/inception.html) | inception_v3|Inception_V3_Weights.DEFAULT| |
| [models.EfficientNet](https://pytorch.org/vision/main/_modules/torchvision/models/efficientnet.html) | efficientnet_b{x}|EfficientNet_B{x}_Weights.DEFAULT| `0` - `7` |
| [models.MobileNetV3](https://pytorch.org/vision/main/_modules/torchvision/models/mobilenetv3.html) | mobilenet_v3_{x}|MobileNet_V3_{x}_Weights.DEFAULT| `large` `small` |
| [models.MobileNetV3](https://pytorch.org/vision/main/_modules/torchvision/models/ShuffleNetV2.html) | shufflenet_v2_x{x}|ShuffleNet_V2_X{x}_Weights.DEFAULT| `0_5` `1_0` `1_5` `2_0` |
| [models.ConvNeXt](https://pytorch.org/vision/main/_modules/torchvision/models/convnext.html) | convnext_{x}|ConvNeXt_{x}_Weights.DEFAULT| `tiny` `small` `base` `large` |
| [models.VisionTransformer](https://pytorch.org/vision/main/models/vision_transformer.html) | vit_{x}|VIT_{x}_Weights.DEFAULT| `b_16` `b_32` `l_16` `l_32` `h_14` |
| [models.SwinTransformer](https://pytorch.org/vision/main/models/swin_transformer.html) | swin_{x}|Swin_{x}_Weights.DEFAULT| `t` `s` `b` |
| [models.SwinTransformer](https://pytorch.org/vision/main/models/swin_transformer.html) | swin_v2_{x}|Swin_V2_{x}_Weights.DEFAULT| `t` `s` `b` |
| `Detection` | | | |
| [models.detection.FasterRCNN](https://pytorch.org/vision/main/_modules/torchvision/models/detection/faster_rcnn.html) | fasterrcnn_{x}|FasterRCNN_{x}_Weights.DEFAULT| `resnet50_fpn` `resnet50_fpn_v2` `mobilenet_v3_large_fpn` `mobilenet_v3_large_320_fpn` |
| [models.detection.RetinaNet](https://pytorch.org/vision/main/_modules/torchvision/models/detection/retinanet.html) | retinanet_{x}|RetinaNet_{x}_Weights.DEFAULT| `resnet50_fpn` `resnet50_fpn_v2`  |
| [models.detection.MaskRCNN](https://pytorch.org/vision/main/_modules/torchvision/models/detection/mask_rcnn.html) | maskrcnn_{x}|MaskRCNN_{x}_Weights.DEFAULT| `resnet50_fpn` `resnet50_fpn_v2`  |
| `Segmentation` | | | |
| [models.segmentation.FCN](https://pytorch.org/vision/main/_modules/torchvision/models/segmentation/fcn.html) | fcn_resnet{x}|FCN_ResNet{x}_Weights.DEFAULT| `50` `101`  |
| [models.segmentation.DeepLabV3](https://pytorch.org/vision/main/_modules/torchvision/models/segmentation/deeplabv3.html) | deeplabv3_{x}|DeepLabV3_{x}_Weights.DEFAULT| `resnet50` `resnet101` `mobilenet_v3_large` |

### Dataset



### Training Interface
Examples of training Wrapping Network can be found in [ultis.py](Modules/ultis.py) and train file [train.py](train.py), we config hyper-parameters in [config.yml](config/new-config.yml) file

- `ultis.py`: Three pre-defined datasets have been established, each serving as a demonstration for the training-testing process of a specific task, [CIFAR10](Modules/ultis.py) for classification, [Lung CT-scan](Modules/ultis.py) for object detection, and [PennFudan](Modules/ultis.py) for binary object segmentation.
- Trainning `train.py`: Our main module is [Model](Modules/train.py) that based on [pytorch-lightning](https://lightning.ai/pages/open-source/) and logged by [neptune-ai](https://neptune.ai/). As shown in the Figure above, we logged hyperparameters, metrics, and results from each run.

![alt text](assets/neptune.jpg)

1. Import the necessary modules:
```python
import yaml
from pytorch_lightning.loggers import NeptuneLogger
from Modules.train import DataModule, Model, get_trainer
```
2. Load Config file and Neptune logging repository:

```Python
with open(args.config_file, 'r') as stream:
        PARAMS = yaml.safe_load(stream)
        print(PARAMS)

neptune_logger = NeptuneLogger(
        api_key="YOUR_API_KEY",
        project ="YOUR_PROJECT_NAME",
        log_model_checkpoints=False,
    )

neptune_logger.log_hyperparams(params=PARAMS)
```
3. Load Data
```Python
# add key and new data class as followed, in Train.py DataModule class
self.data_class = {
   "CIFAR10": CIFAR10read,
   "LungCT-Scan": LungCTscan,
   "Dubai": DubaiAerialread,
   "PennFudan": PennFudanDataset,

   # New dataset
   "New Dataset": NeWDatasetClass
}
```
```Python
data = DataModule(PARAMS['dataset_settings'], PARAMS['training_settings'], [None, None])
```
4. Fine tune and evaluate moodel
```Python
model = Model(PARAMS=PARAMS)
trainer = get_trainer(PARAMS['training_settings'], neptune_logger)
# train
trainer.fit(model, data)
# test
trainer.test(model, data)
```
5. Run training in environment
```bash
python Modules/train.py -c CONFIG_FILE
```
6. Run training by Docker
```bash
docker run --rm -it --init \
  --gpus=all \
  --ipc=host \
  --volume="$PWD:/app" \
  pytorch-finetune:runtime python Modules/train.py
```
#

## Testing Interface
We deploy (demo) our model using [Gradio](https://gradio.app/), which supports to visualize results from 3 tasks: classification, detection, and segmentation, depending on the selected model.

![alt text](assets/gradio.png)
1. Create a folder named `models` and save all checkpoints inside it. 
2. Run app
- by environment: 
```bash
python Modules/app.py
```
- by Docker:
```bash
docker run --rm -it --init \
  --gpus=all \
  --ipc=host \
  --volume="$PWD:/app" \
  -p 7860:7860 \
  pytorch-finetune:runtime python Modules/app.py
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
| `logger` | neptune account |  |  |
| project | your project | logger |str  |
| api_key | your account token | logger |str  |
| tags | Runtime Tags | logger |[str]  |
| task | task of experiment |  | str: "classification", "detection", "segmentation" |
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
| loss | loss function  | training_settings  | str: "ce" (classification/segmentation), "dice", "mse", "none"(detection) |
| metric | metric name  | training_settings  | str: "accuracy", "dice", "mAP" |
| ckpt_path | path to check-points  | training_settings  | str |
| n_epoch | num epoch  | training_settings  | int |
| n_batch | batch size  | training_settings  | int |
| num_workers | num workers to dataloader | training_settings  | int |
| optimizer | optimizer | training_settings  | str: "adam", "sgd" |
| lr_scheduler | learning rate scheduler | training_settings  | str: "step", "multistep", "reduce_on_plateau" |
| early_stopping | early stopping | training_settings  | bool |
| lr | learning rate | training_settings  | float|
| lr_step | learning rate step for decay| training_settings  | int|
| lr_decay | learning rate decay rate | training_settings  | float|
| momentum | momentum for optimizer | training_settings  | float|
| weight_decay | weight decay for "sgd" | training_settings  | float|
</tr>
The understanding of the functioning of the configuration file is best obtained by referring to the actual "config.yaml" file. It is crucial to acknowledge that the module's flexibility is maintained by avoiding excessive hard coding of parameters. This is because fine-tuning and optimizing the parameters play a vital role in the development process. As part of our ongoing efforts to enhance the performance of the module, we anticipate making further refinements to the configurations. Consequently, it is likely that the config file will be subject to future updates to reflect these optimizations.

# Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

When contributing to this project, please adhere to the following guidelines:

- Fork the repository and create a branch for your feature or bug fix.
- Ensure that your code is well-documented and follows the project's coding conventions.
- Write clear commit messages and provide a detailed description of your changes.
- Ensure that your code passes all existing tests and write new tests when applicable.
# License
This project is licensed under the MIT License.

# Acknowledgments
- AISeed Inc for supporting developers.
- The Pytroch/ PyTorch Lightning community for providing a powerful and flexible deep learning framework.
- The Neptune Ai community for providing a professional logger.
# Contact
For any questions or inquiries, please contact dtvu1707@gmail.com
