import torch.nn as nn

class BaseModel(nn.Module):
    """
    Base model for all models, which has 3 main methods:
    - __init__: initialize model
    - forward: forward pass
    - _model_selection: select model from pretrained model zoo
    """
    def __init__(self, 
                name,
                model,
                weight=None,
                is_freeze=True,
                **kwargs):
        super(BaseModel, self).__init__()

        self.backbone_name = name
        if weight is not None:
            self.meta = weight.meta
            self.preprocess = weight.transforms()
        self.model = model(weights=weight)

        if is_freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.freeze_parameters(self.model, is_freeze)

    def freeze_parameters(self, model, is_freeze):
        if is_freeze:
            _ = list(map(lambda param: param.requires_grad_(False), model.parameters()))

    def _model_type_check(self, types):
        assert any(map(lambda t: isinstance(self.model, t), types)), \
            "Model type not found in %s" % types

    def forward(self, x, y=None):
        raise NotImplementedError()
    
    def _model_selection(self, **kwargs):
        raise NotImplementedError()
      

        
        


        