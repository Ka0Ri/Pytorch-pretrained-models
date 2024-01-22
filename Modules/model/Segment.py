import torch.nn as nn
import torch
from torchvision.models.segmentation import FCN, DeepLabV3
import Modules.Model.base_model as base_model                                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
def convrelu(in_channels, out_channels, kernel, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding=padding),
        nn.ReLU(inplace=True),
    )

def deconvrelu(in_channels, out_channels, kernel, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding, output_padding=1),
        nn.ReLU(inplace=True),
    )

class ConvUNet(nn.Module):
    """
    Convolutional UNet module
    
    """
    def __init__(self, model_configs):
        super(ConvUNet, self).__init__()

        n_filters = model_configs["n_filters"]
        # Contracting Path
        self.n_layer = model_configs['n_layers']
        conv = model_configs['conv']


        self.downsamling_layers = nn.ModuleList([nn.Sequential(convrelu(model_configs["channel"], n_filters, conv["k"], 1, conv["p"]), 
                                                # nn.MaxPool2d(kernel_size=2)
                                                )])
        for i in range(self.n_layer - 1):
            self.downsamling_layers.append(nn.Sequential(
                # nn.MaxPool2d(kernel_size=2),
                convrelu(n_filters * (2 ** i), n_filters * (2 ** (i + 1)), conv["k"], conv["s"], conv["p"]),
                )
            )

        transpose_conv = model_configs['transpose_conv']
        self.upsampling_layers = nn.ModuleList()
        for i in range(self.n_layer - 1, 1, -1):
            self.upsampling_layers.append(deconvrelu(n_filters * (2 ** i), n_filters * (2 ** (i - 1)), transpose_conv["k"], transpose_conv["s"], transpose_conv["p"]))
            self.upsampling_layers.append(convrelu(n_filters * (2 ** i), n_filters * (2 ** (i - 1)), conv["k"], 1, conv["p"]))

        # Expansive Path
        self.upsampling_layers.append(deconvrelu(2 * n_filters, n_filters, transpose_conv["k"], transpose_conv["s"], transpose_conv["p"]))
        self.upsampling_layers.append(nn.Conv2d(2 * n_filters, model_configs["channel"], conv["k"], padding='same'))
        # self.out = nn.Conv2d(n_filters, model_configs["channel"], conv["k"], padding='same')
       

    def forward(self, x, y=None):

        encode = [x]
        for i in range(self.n_layer):
            down = self.downsamling_layers[i](encode[-1])
            print("down", down.shape)
            encode.append(down)

        up = encode[-1]
       
        for i in range(0, 2 * self.n_layer - 2, 2):
            up = self.upsampling_layers[i](up)
            print("up", up.shape)
            cat = torch.cat([up, encode[-(i//2) - 2]], dim=1)
            print("cat", cat.shape)
            up = self.upsampling_layers[i+1](cat)
            print("up", up.shape)
            
        return up

class SegmentModel(base_model.BaseModel):
    '''
    Wrapping Segmentation model with customized head, 
    The given models can be used in reference mode (is_full = True)
    with pretrained weights (is_pretrained = True)
    or fine tuning (is_freeze = False) with explicit the number of classes 

    - Supported models: fcn, deeplab [-option m: medium, l: large]
   
    '''

    def __init__(self, 
                model,
                weight=None,
                is_freeze=True,
                is_full=False,
                n_cls=2,
                **kwargs):
        super().__init__(model, weight, is_freeze)

        self._model_selection(is_full, n_cls)
        
    def _model_selection(self, is_full, n_cls):
        
        self._model_type_check([FCN, DeepLabV3])
        # Modify last layer
        if not is_full:
            if isinstance(self.model, (FCN, DeepLabV3)):
                self.model.aux_classifier = None
                num_ftrs = self.model.classifier[-1].in_channels
                self.model.classifier[-1] = nn.Conv2d(num_ftrs, n_cls, 1)

    def forward(self, x):
        return self.model(x)['out']


