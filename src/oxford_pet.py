import os
import torch
import shutil
import numpy as np

from PIL import Image
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from tqdm import tqdm
from urllib.request import urlretrieve

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        if self.transform is not None:
            sample = self.transform(**sample)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

class trainTransform():
    def __init__(self):
        pass
    def __call__(self, image,mask,trimap):
        return {
            "image":torch.Tensor(image/255-0.5).to(torch.float),
            "mask":mask,
            'trimap': trimap
        }

def load_dataset(data_path, mode, batchsize):
    # implement the load dataset function here
    print(f"loading dataset... ({mode})")
    transform=trainTransform()
    dataset=SimpleOxfordPetDataset(root=data_path,mode=mode,transform=transform)
    dataloader=torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=(mode=="train"), pin_memory=True)
    num_batch=len(dataloader)
    print(f"dataset ready! (batchsize={batchsize})")
    return dataloader,num_batch
    n_samples=len(dataset)
    img_dataloader=[]
    mask_dataloader=[]
    for i in range(0,n_samples,batchsize):
        img_batch=[]
        mask_batch=[]
        for j in range(i,min(i+batchsize,n_samples)):
            img_batch.append(dataset[j]["image"])
            mask_batch.append(dataset[j]["mask"])
        img_batch=np.array(img_batch)
        mask_batch=np.array(mask_batch)
        img_dataloader.append(torch.tensor(img_batch).to(torch.float32)/255-0.5)
        mask_dataloader.append(torch.tensor(mask_batch))
    num_batch=len(img_dataloader)
    print(f"dataset ready!(batchsize={batchsize})")
    return img_dataloader,mask_dataloader,num_batch
    assert False, "Not implemented yet!"

def test_for_dataset():
    dataloader,num_batch = load_dataset(data_path="dataset/oxford-iiit-pet", mode="train", batchsize=128)
    for batch in dataloader:
        images = batch["image"]
        print(images[0])
        mask = batch["mask"]
        print(mask[0])
        break
    for batch in dataloader:
        images = batch["image"]
        print(images.shape)

if __name__=="__main__":
    test_for_dataset()