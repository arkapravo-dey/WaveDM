import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from PIL import Image
import pdb
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.models.vgg as vgg
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
from models.model_dense import *
from models.arch import HFRM
from datasets.dataset import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import shutil

def BatchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean(dim=(1,2,3)).sqrt()
    ps = 20 * torch.log10(1/rmse)
    return ps

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def compute_l1_loss(input, output):
    return torch.mean(torch.abs(input-output))
        
def loss_Textures(x, y, nc=3, alpha=1.2, margin=0):
    xi = x.contiguous().view(x.size(0), -1, nc, x.size(2), x.size(3))
    yi = y.contiguous().view(y.size(0), -1, nc, y.size(2), y.size(3))
    xi2 = torch.sum(xi * xi, dim=2)
    yi2 = torch.sum(yi * yi, dim=2)
    out = nn.functional.relu(yi2.mul(alpha) - xi2 + margin)
    return torch.mean(out)

class LossNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '13': "relu3",
            '22': "relu4",
            '31': "relu5",
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return output

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

# [Your existing imports remain unchanged]

# -----------------------------------------
# Start of main script logic
# -----------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=5)
parser.add_argument('--dataset_name', type=str, default="raindrop")
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--b1', type=float, default=0.5)
parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--decay_epoch', type=int, default=40)
parser.add_argument('--n_cpu', type=int, default=8)
parser.add_argument('--img_height', type=int, default=256)
parser.add_argument('--img_width', type=int, default=256)
parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--sample_interval', type=int, default=500)
parser.add_argument('--checkpoint_interval', type=int, default=-1)
parser.add_argument('--mse_avg', action='store_true')
parser.add_argument('--data_url', type=str, default="")
parser.add_argument('--init_method', type=str, default="")
parser.add_argument('--train_url', type=str, default="")
opt = parser.parse_args()
print(opt)

save_dir = f'/kaggle/working/saved_models/{opt.dataset_name}'
os.makedirs(f'images/{opt.dataset_name}', exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
print(f"✅ Save directory created: {save_dir}, Exists: {os.path.exists(save_dir)}")

cuda = torch.cuda.is_available()
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
tvloss = TVLoss()
lossmse = torch.nn.MSELoss()
lambda_pixel = 100
patch = (1, opt.img_height//2**4, opt.img_width//2**4)

img_channel = 3
dim = 32
enc_blks = [2, 2, 2, 4]
middle_blk_num = 6
dec_blks = [2, 2, 2, 2]
generator = HFRM(in_channel=img_channel, dim=dim, mid_blk_num=middle_blk_num, enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
print("Total_params_model: {:.2f}M".format(sum(p.numel() for p in generator.parameters() if p.requires_grad)/1e6))

if cuda:
    generator = generator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    lossnet = LossNetwork().float().cuda()

if opt.epoch != 0:
    generator.load_state_dict(torch.load(f'{save_dir}/best.pth'))
else:
    generator.apply(weights_init_normal)

generator = nn.DataParallel(generator)
device = torch.device("cuda:0")
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

mytransform = transforms.Compose([transforms.ToTensor()])
data_root = '/kaggle/working/WaveDM/data/raindrop/train'
myfolder = myImageFloder(root=data_root, transform=mytransform, crop=False, resize=False, crop_size=480, resize_size=480)
dataloader = DataLoader(myfolder, num_workers=opt.n_cpu, batch_size=opt.batch_size, shuffle=True)
print('✅ Data loader ready.')

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

best_psnr = 0
step = 0

try:
    for epoch in range(opt.epoch, opt.n_epochs):
        epoch_psnr = []
        for i, batch in enumerate(tqdm(dataloader), 0):
            step += 1
            current_lr = opt.lr * (1 / 2) ** (step / 100000)
            for param_group in optimizer_G.param_groups:
                param_group["lr"] = current_lr

            real_A, real_B = Variable(batch[0].cuda()), Variable(batch[1].cuda())

            optimizer_G.zero_grad()
            fake_B = generator(real_A)
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            loss_p = compute_l1_loss(fake_B * 255, real_B * 255) * 2
            loss_G = loss_p
            loss_G.backward()
            optimizer_G.step()

            psnr = BatchPSNR(fake_B, real_B)
            epoch_psnr.append(psnr.mean().item())

            if i % 100 == 0:
                print(f"Epoch {epoch} Batch {i} - G loss: {loss_G.item():.4f}, Pixel Loss: {loss_pixel.item():.4f}")

        epoch_avg_psnr = np.mean(epoch_psnr)
        print(f"📈 Epoch {epoch} PSNR: {epoch_avg_psnr:.4f} | Best so far: {best_psnr:.4f}")

        # === Save Paths ===
        epoch_model_path = os.path.join(save_dir, f'epoch_{epoch}.pth')
        latest_model_path = os.path.join(save_dir, 'latest.pth')
        best_model_path = os.path.join(save_dir, 'best.pth')

        # === Save models ===
        torch.save(generator.module.state_dict(), epoch_model_path)
        torch.save(generator.module.state_dict(), latest_model_path)
        print(f"✅ Saved model: {epoch_model_path}")
        print(f"✅ Saved latest model: {latest_model_path}")

        # Save best model
        if epoch_avg_psnr > best_psnr:
            best_psnr = epoch_avg_psnr
            torch.save(generator.module.state_dict(), best_model_path)
            print(f"🏆 New best model saved: {best_model_path}")

        # Also copy to working directory for export
        shutil.copy(epoch_model_path, f'/kaggle/working/epoch_{epoch}.pth')
        shutil.copy(latest_model_path, '/kaggle/working/latest.pth')
        shutil.copy(best_model_path, '/kaggle/working/best.pth')

        # List saved files
        print("📁 Saved models in /kaggle/working/:", os.listdir('/kaggle/working/'))

except Exception as e:
    print(f"❌ Training crashed due to: {e}")
    crash_path = f'{save_dir}/crash_epoch_{epoch}.pth'
    torch.save(generator.module.state_dict(), crash_path)
    print(f"💾 Crash backup saved to: {crash_path}")
    raise e
