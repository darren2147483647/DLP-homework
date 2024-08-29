import os
from argparse import ArgumentParser

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from dataset import dataset_
from ddpm import DDPM
from evaluator import evaluation_model
from torchvision.utils import make_grid, save_image

os.makedirs('result', exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_scheduler(timesteps):
    return DDPMScheduler(num_train_timesteps=timesteps, beta_schedule='squaredcos_cap_v2')

def load_model(ckpt):
    model = DDPM().to(device)
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def inference(dataloader, noise_scheduler, timesteps, model, eval_model, save_prefix='test'):
    all_results = []
    acc = []
    progress_bar = tqdm(dataloader)
    for idx, y in enumerate(progress_bar):
        y = y.to(device)
        x = torch.randn(1, 3, 64, 64).to(device)
        denoising_result = []
        for i, t in enumerate(noise_scheduler.timesteps):
            with torch.no_grad():
                residual = model(x, t, y)

            x = noise_scheduler.step(residual, t, x).prev_sample
            if i % (timesteps // 10) == 0:
                denoising_result.append(x.squeeze(0))

        acc.append(eval_model.eval(x, y))
        progress_bar.set_postfix_str(f'image: {idx}, accuracy: {acc[-1]:.4f}')

        denoising_result.append(x.squeeze(0))
        denoising_result = torch.stack(denoising_result)
        row_image = make_grid((denoising_result + 1) / 2, nrow=denoising_result.shape[0], pad_value=0)
        save_image(row_image, f'result/{save_prefix}_{idx}.png')
        
        all_results.append(x.squeeze(0))
    all_results = torch.stack(all_results)
    all_results = make_grid(all_results, nrow=8)
    save_image((all_results + 1) / 2, f'result/{save_prefix}_result.png')
    return acc


if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

    os.makedirs('result', exist_ok=True)
    ckpt = 'checkpoints/checkpoint_100.pth'
    timesteps = 20
    model = load_model(ckpt)
    noise_scheduler = get_scheduler(timesteps)
    eval_model = evaluation_model()
    
    test_loader = DataLoader(dataset_("iclevr", "test"))
    new_test_loader = DataLoader(dataset_("iclevr", "new_test"))
    test_acc = inference(test_loader, noise_scheduler, timesteps, model, eval_model, 'test')
    new_test_acc = inference(new_test_loader, noise_scheduler, timesteps, model, eval_model, 'new_test')
    print(f'test accuracy: {np.mean(test_acc)}')
    print(f'new test accuracy: {np.mean(new_test_acc)}')
    label_set_test_loader = DataLoader(dataset_("iclevr", "label_set"))
    inference(label_set_test_loader, noise_scheduler, timesteps, model, eval_model, 'label_set')