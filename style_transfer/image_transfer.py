import os
import torch
from PIL import Image
from torchvision import transforms
import torchvision.utils as utils
import numpy as np
from .models.RevResNet import RevResNet
from .models.cWCT import cWCT

def img_resize(img, max_size, down_scale=None):
    w, h = img.size

    if max(w, h) > max_size:
        w = int(1.0 * img.size[0] / max(img.size) * max_size)
        h = int(1.0 * img.size[1] / max(img.size) * max_size)
        img = img.resize((w, h), Image.BICUBIC)
    if down_scale is not None:
        w = w // down_scale * down_scale
        h = h // down_scale * down_scale
        img = img.resize((w, h), Image.BICUBIC)
    return img

def load_segment(image_path, size=None):
    def change_seg(seg):
        color_dict = {
            (0, 0, 255): 3,  # blue
            (0, 255, 0): 2,  # green
            (0, 0, 0): 0,  # black
            (255, 255, 255): 1,  # white
            (255, 0, 0): 4,  # red
            (255, 255, 0): 5,  # yellow
            (128, 128, 128): 6,  # grey
            (0, 255, 255): 7,  # lightblue
            (255, 0, 255): 8  # purple
        }
        arr_seg = np.array(seg)
        new_seg = np.zeros(arr_seg.shape[:-1])
        for x in range(arr_seg.shape[0]):
            for y in range(arr_seg.shape[1]):
                if tuple(arr_seg[x, y, :]) in color_dict:
                    new_seg[x, y] = color_dict[tuple(arr_seg[x, y, :])]
                else:
                    min_dist_index = 0
                    min_dist = 99999
                    for key in color_dict:
                        dist = np.sum(np.abs(np.asarray(key) - arr_seg[x, y, :]))
                        if dist < min_dist:
                            min_dist = dist
                            min_dist_index = color_dict[key]
                        elif dist == min_dist:
                            try:
                                min_dist_index = new_seg[x, y - 1, :]
                            except Exception:
                                pass
                    new_seg[x, y] = min_dist_index
        return new_seg.astype(np.uint8)

    if not os.path.exists(image_path):
        print("Can not find image path: %s " % image_path)
        return None

    image = Image.open(image_path).convert("RGB")

    if size is not None:
        w, h = size
        transform = transforms.Resize((h, w), interpolation=Image.NEAREST)
        image = transform(image)

    image = np.array(image)
    if len(image.shape) == 3:
        image = change_seg(image)
    return image


def get_image(content_path, style_path):
    ckpt_path = './checkpoints/art_image.pt'
    RevNetwork = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256], in_channel=3, mult=4, hidden_dim=64, sp_steps=1)
    device = torch.device('cpu')
    state_dict = torch.load(ckpt_path, map_location=device)
    RevNetwork.load_state_dict(state_dict['state_dict'])
    RevNetwork = RevNetwork.to(device)
    RevNetwork.eval()
    cwct = cWCT()

    content = Image.open(content_path).convert('RGB')
    style = Image.open(style_path).convert('RGB')

    ori_csize = content.size

    content = img_resize(content, 1280, down_scale=RevNetwork.down_scale)
    style = img_resize(style, 1280, down_scale=RevNetwork.down_scale)
    content = transforms.ToTensor()(content).unsqueeze(0).to(device)
    style = transforms.ToTensor()(style).unsqueeze(0).to(device)

    # Stylization
    with torch.no_grad():
        # Forward inference
        z_c = RevNetwork(content, forward=True)
        z_s = RevNetwork(style, forward=True)

        # Transfer
        z_cs = cwct.transfer(z_c, z_s, None, None)

        # Backward inference
        stylized = RevNetwork(z_cs, forward=False)
    # return stylized
    grid = utils.make_grid(stylized.data, nrow=1, padding=0)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    out_img = Image.fromarray(ndarr)
    return out_img
