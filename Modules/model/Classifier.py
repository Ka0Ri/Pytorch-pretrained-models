import torch.nn as nn
from torchvision.models import ResNet, DenseNet, EfficientNet, VGG, MobileNetV3, Inception3,\
                                ShuffleNetV2, ConvNeXt, VisionTransformer, SwinTransformer  
                                
import Modules.model.base_model as base_model
         
class ClassificationModel(base_model.BaseModel):
    '''
    Wrapping Classification with customized headers
    The given models can be used in reference mode (is_full = True)
    with pretrained weights (is_pretrained = True)
    or fine tuning (is_freeze = False) with explicit the number of classes
    - Supported models: resnet, shuffle, inception, resnext, vgg, mobile, , 
                        vit, swin, wideresnet, dense, convnext, efficient
    [-option s: small, m: medium, l: large]
    '''

    def __init__(self, 
                name,
                model,
                weight=None,
                is_freeze=True,
                is_full=False,
                n_cls=2):
        super().__init__(name, model, weight, is_freeze)

        self._model_selection(is_full, n_cls)

    def _model_selection(self, is_full, n_cls):

        self._model_type_check([ResNet, DenseNet, EfficientNet, VGG, MobileNetV3, Inception3, \
                                ShuffleNetV2, ConvNeXt, VisionTransformer, SwinTransformer])
        
        # Modify last layer
        if not is_full:
            # Remove Classification head
            if isinstance(self.model, (ResNet, ShuffleNetV2, Inception3)):
                if isinstance(self.model, Inception3): 
                    self.model.aux_logits=False
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Identity()
            
            elif isinstance(self.model, (VGG, MobileNetV3, EfficientNet)):

                if isinstance(self.model, VGG): 
                    num_ftrs = self.model.features[-4].out_channels
                else:
                    num_ftrs = self.model.features[-1].out_channels
                self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.model.classifier = nn.Identity()

            elif isinstance(self.model, (DenseNet, ConvNeXt)):
                if isinstance(self.model, ConvNeXt): 
                    num_ftrs = self.model.classifier[2].in_features
                else:
                    num_ftrs = self.model.classifier.in_features
                self.model.classifier = nn.Identity()

            elif isinstance(self.model, VisionTransformer):
                num_ftrs = self.model.head.in_features
                self.model.head = nn.Identity()

            elif isinstance(self.model, SwinTransformer):
                num_ftrs = self.model.head.in_features
                self.model.head = nn.Identity()

            self.model = nn.Sequential(self.model,
                                       nn.Flatten(start_dim=1),
                                       nn.Linear(num_ftrs, n_cls))

    def forward(self, x, y=None):
        return self.model(x)