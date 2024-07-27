import matplotlib.pyplot as plt
import torch
import numpy as np
def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    return 2*(pred_mask*gt_mask).sum()/((pred_mask+gt_mask).sum()+1e8)
    assert False, "Not implemented yet!"

def show_image(image):
    if image.shape==(3, 256, 256):
        image = np.transpose(image, (1, 2, 0))
    assert image.shape == (256, 256, 3)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def show_mask(mask):
    if mask.shape==(1, 256, 256):
        mask = np.transpose(mask, (1, 2, 0))
    assert mask.shape == (256, 256, 1)
    plt.imshow(mask)
    plt.axis('off')
    plt.show()

def save(model,file_name="saved_models/weight.pt"):
    print(f"=>save {file_name}")
    torch.save(model.state_dict(),file_name)

def load(model,file_name="saved_models/weight.pt"):
    print(f"<=load {file_name}")
    model.load_state_dict(torch.load(file_name))
