from train import Model
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import os
from torchvision import transforms
import gradio as gr

with open("models/class_name.txt", "r", encoding='utf-8') as f:
    class_names = f.read().splitlines()

PARAMS =  {
    "architect_settings" : {
            "task": "None",
            "name": "model-test",
            "backbone": {
                    "name": "selectnet-s",
                    "is_full": False,
                    "is_pretrained": True,
                    "is_freeze": False, 
            },
            "n_cls": 2,
            },
    "dataset_settings": {
            
            },
    "training_settings":{
    
    }
}

model_list = [name.split('.')[0] for name in os.listdir("models") if name.endswith('ckpt')]
model_list += ["fcn-m", "resnet-s", "fasterrcnn-s", "maskrcnn-s"]
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

def predict(image, model_choice):
    labels, detection = None, None
    if(model_choice in ["fcn-m", "resnet-s", "fasterrcnn-s", "maskrcnn-s"]):
        PARAMS['architect_settings']['backbone']['is_full'] = True
        PARAMS['architect_settings']['backbone']['name'] = model_choice
        if(model_choice in ["fasterrcnn-s", "maskrcnn-s"]):
            PARAMS['task'] = 'detection'
        elif(model_choice == "fcn-m"):
            PARAMS['task'] = 'segmentation'
        else:
            PARAMS['task'] = 'classification'
        model = Model(PARAMS)
    else:
        model = Model.load_from_checkpoint(f"models/{model_choice}.ckpt").cpu()
    model.eval()
    transforms = model.model.preprocess
    tensor_image = transforms(image)
    with torch.no_grad():
        y_hat = model(tensor_image.unsqueeze(0))
        if(model.task == "classification"):
            preds = torch.softmax(y_hat, dim=-1).tolist()
            labels = {class_names[k]: float(v) for k, v in enumerate(preds[0][:-1])}
        elif(model.task == "segmentation"):
            num_classes = y_hat.shape[1]
            masks = y_hat[0]
            classes_masks = masks.argmax(0) == torch.arange(num_classes)[:, None, None]
            tensor_image = inv_normalize(tensor_image)
            detection = draw_segmentation_masks((tensor_image * 255.).to(torch.uint8), 
                                              masks=classes_masks, alpha=.6)
            detection = detection.numpy().transpose(1, 2, 0) / 255.
        elif(model.task == "detection"):
            if("maskrcnn" in model_choice):
                boolean_masks = [out['masks'][out['scores'] > .75] > 0.5
                                for out in y_hat][0]
                detection = draw_segmentation_masks((tensor_image * 255.).to(torch.uint8),
                                                    boolean_masks.squeeze(1), alpha=0.8)
            else:
                detection = draw_bounding_boxes((tensor_image * 255.).to(torch.uint8), 
                                                    boxes=y_hat[0]["boxes"][:5],
                                                    colors="red",
                                                    width=5)
            detection = detection.numpy().transpose(1, 2, 0) / 255.

    return labels, detection

title = "Application Demo "
description = "# A Demo of Wrapping Pretrained Networks"
example_list = [["examples/" + example] for example in os.listdir("examples")]

with gr.Blocks() as demo:
    demo.title = title
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            model = gr.Dropdown(model_list, label="Select Model", interactive=True)
            im = gr.Image(type="pil", label="input image")
            label_conv = gr.Label(label="Predictions", num_top_classes=4)
        with gr.Column():
            im_detection = gr.Image(type="pil", label="Detection")
            btn = gr.Button(value="predict")
    btn.click(predict, inputs=[im, model], outputs=[label_conv, im_detection])
    gr.Examples(examples=example_list, inputs=[im, model], outputs=[label_conv, im_detection])
      

if __name__ == "__main__":
    demo.launch(share=False)