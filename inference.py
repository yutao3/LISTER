import os  
import argparse as arg  
import time
import torch
import torch.nn as nn
import numpy as np  
import cv2   
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from PIL import Image   
from glob import glob
from torchvision import transforms

from DenseNet161UNet.model import DenseDepth
from ViTABUNet.model import UnetAdaptiveBins
from ViTABUNet import model_io
from NWFCCRF.model import NewCRFDepth

def normalise(value):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() #if vmin is None else vmin 
    vmax = value.max() #if vmax is None else vmax   
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0  
    return value
    
def image_loader(image_name):
    # load image, returns cuda tensor
    image = Image.open(image_name)
    width, height = image.size
    channels = len(image.getbands())
    # Check if the image has the required dimensions
    if width != 640 or height != 480:
        raise ValueError(f'Image {image_name} does not have the correct size of 640x480, but instead has size {width}x{height}')
    # Check if the image is single channel, if so convert it to 3-channel
    if channels == 1:
        image = image.convert("RGB")
    loader = transforms.Compose([transforms.ToTensor()])
    image = loader(image).float()
    image = image.unsqueeze(0)
    return image

def predict_image(model, image, args):
    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    with torch.no_grad():
        image = image.to(device)
        if args.model == "D" or args.model == "N":
            pred = model(image)
        elif args.model == "V":
            _, pred, _ = model(image)
        pred = pred.cpu()
        final = nn.functional.interpolate(pred, image.shape[-2:], mode='bilinear', align_corners=True)
        return final
    
def save_output_image(pred, out_file):
    pred = pred.astype(np.float32)
    Image.fromarray(pred).save(out_file, mode='F')

def main():
    parser = arg.ArgumentParser(description="Inference Process")
    parser.add_argument("--model", "-m", type=str, default="D", help="D for DenseNet-161-U-Net; V for ViT-AB-U-Net; N for NW-FC-CRF")
    parser.add_argument("--weights", "-w", type=str, help="path to pretrained model")
    parser.add_argument("--device", "-d", type=str, default="cuda")
    parser.add_argument("--data", type=str, default="test_data/", help="Path to input tiles")
    args = parser.parse_args()

    if len(args.weights) and not os.path.isfile(args.weights):
        raise FileNotFoundError("{} no such file".format(args.weights))    

    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    print("Using device: {}".format(device))

    # Get Test Images  
    img_list = glob(args.data+"*.tif")

    # Initialise the model and load the pretrained weights 
    if args.model == "D":
        model = DenseDepth(encoder_pretrained=False)
        ckpt = torch.load(args.weights)
        model.load_state_dict(ckpt["model_state_dict"])
        print("Successfully loaded the pre-trained weights for the DenseNet-161-U-Net model.")
        
    elif args.model == "V":
        model = UnetAdaptiveBins.build(n_bins=256, min_val=1, max_val=254, norm='linear').to(device)
        model = model_io.load_checkpoint(args.weights, model)[0]
        print("Successfully loaded the pre-trained weights for the ViT-AB-U-Net model.")
        
    elif args.model == "N":
        model = NewCRFDepth(version='large07', inv_depth=False, max_depth=255)
        ckpt = torch.load(args.weights)
        model.load_state_dict(ckpt['model'])
        print("Successfully loaded the pre-trained weights for the NW-FC-CRF model.")
        
    else: 
        print("Please specify which model to use from D, V and N! D for DenseNet-161-U-Net; V for ViT-AB-U-Net; N for NW-FC-CRF")
        
    model.eval()
    model.cuda()

    # Begin inference loop 
    print("Begin DTM inference loop ...")
   
    for idx, img_name in enumerate(img_list):
        img = image_loader(img_name).to(device)
        prediction = predict_image(model, img, args)
        prediction = normalise(prediction)
        prediction = prediction.squeeze()
                
        base, ext = os.path.splitext(img_name)
        new_img_name = base + "_result" + ext
        save_output_image(prediction, new_img_name)

if __name__ == "__main__":
    print("Using torch version: ", torch.__version__)
    main()
