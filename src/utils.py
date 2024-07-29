import matplotlib.pyplot as plt
import torch
import numpy as np
def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    return 2*(pred_mask*gt_mask).sum()/((pred_mask+gt_mask).sum()+1e-8)
    assert False, "Not implemented yet!"

def show_image(image,fix=True):
    image=image.cpu()
    if fix:
        image=image+0.5
    if image.shape==(3, 256, 256):
        image = np.transpose(image, (1, 2, 0))
    assert image.shape == (256, 256, 3)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def show_mask(mask):
    mask=mask.cpu()
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

if __name__=="__main__":
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.device_count())  # 应显示你的 GPU 数量
    print(torch.cuda.get_device_name(0))