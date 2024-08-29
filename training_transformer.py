import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self):
        self.model.train()
        total_loss=0
        for batch in train_loader:
            batch=batch.to(device=args.device)
            logits, z_indices = self.model.forward(batch)
            loss=F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            total_loss+=loss.item()
        return total_loss/(len(train_loader)+1e-8)
        pass

    def eval_one_epoch(self):
        self.model.eval()
        with torch.no_grad():
            total_loss=0
            for batch in val_loader:
                batch=batch.to(device=args.device)
                logits, z_indices = self.model.forward(batch)
                loss=F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))
                total_loss+=loss.item()
            return total_loss/(len(val_loader)+1e-8)
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.transformer.parameters(),lr=args.learning_rate)
        scheduler = None
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=5, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    parser.add_argument('--nam', type=str, default='mkgit', help='savefile name')

    args = parser.parse_args()

    seed=1 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    best_train_loss=100
    best_val_loss=100
    best_train=0
    best_val=0
    train_loss_his=[]
    val_loss_his=[]
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = train_transformer.train_one_epoch()
        val_loss = train_transformer.eval_one_epoch()
        train_loss_his.append(train_loss)
        val_loss_his.append(val_loss)
        if epoch%args.save_per_epoch==0:
            file_name=f"{args.nam}_{epoch}.pt"
            print(f"epoch:{epoch}|train_loss:{train_loss:.4f}|val_loss:{val_loss:.4f}|save:{file_name}")
            torch.save(train_transformer.model.transformer.state_dict(),os.path.join("pt",file_name))
        else:
            print(f"epoch:{epoch}|train_loss:{train_loss:.4f}|val_loss:{val_loss:.4f}")
        if best_train_loss>train_loss:
            best_train_loss=train_loss
            best_train=epoch
        if best_val_loss>val_loss:
            best_val_loss=val_loss
            best_val=epoch
    print(f"best train|epoch:{best_train}|loss:{best_train_loss}")
    print(f"best val|epoch:{best_val}|loss:{best_val_loss}")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_his,label="train",color="red")
    plt.plot(val_loss_his,label="val",color="blue")
    plt.title(f"loss per epoch")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True)
    plt.show()
    #pass