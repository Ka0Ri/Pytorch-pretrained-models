from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.ssd import SSDHead, SSD
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import _utils
from torchvision.models.detection import FasterRCNN, RetinaNet, MaskRCNN
import Modules.Model.base_model as base_model

     
class DetectionModel(base_model.BaseModel):
    '''
    Wrapping Detection model with customized head,
    The given models can be used in reference mode (is_full = True)
    with pretrained weights (is_pretrained = True)
    or fine tuning (is_freeze = False) with explicit the number of classes 

    - Supported models: fasterrcnn, ssd, retinanet
    
    '''

    def __init__(self,
                model,
                weight=None,
                is_freeze=True,
                is_full=False,
                n_cls=2,
                **kwargs) -> None:
        super().__init__(model, weight, is_freeze)

        self._model_selection(is_full, n_cls)

    def _model_selection(self, is_full, n_cls):
        
        self._model_type_check([FasterRCNN, RetinaNet, SSD, MaskRCNN])
       
        if not is_full:
        # Modify detection head
            if isinstance(self.model, FasterRCNN):
                num_ftrs = self.model.roi_heads.box_predictor.cls_score.in_features
                self.model.roi_heads.box_predictor = FastRCNNPredictor(num_ftrs, n_cls)   
            elif isinstance(self.model, RetinaNet):
                num_ftrs = self.model.backbone.out_channels
                num_anchors = self.model.anchor_generator.num_anchors_per_location()
                self.model.head = RetinaNetHead(num_ftrs, num_anchors[0], n_cls)
            elif isinstance(self.model, SSD):
                num_ftrs = _utils.retrieve_out_channels(self.model.backbone, (320, 320))
                num_anchors = self.model.anchor_generator.num_anchors_per_location()
                self.model.head = SSDHead(num_ftrs, num_anchors, n_cls)
            elif isinstance(self.model, MaskRCNN):
                num_ftrs = self.model.roi_heads.box_predictor.cls_score.in_features
                self.model.roi_heads.box_predictor = FastRCNNPredictor(num_ftrs, n_cls)
                mask_num_ftrs = self.model.roi_heads.mask_predictor[0].in_channels
                self.model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_num_ftrs, 256, n_cls)

    def forward(self, x, y=None):

        images = list(image for image in x)
        if y is not None:
            targets = [{k: v for k, v in t.items()} for t in y]
            return self.model(images, targets)
        else:
            return self.model(images)