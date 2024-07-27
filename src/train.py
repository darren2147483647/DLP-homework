import argparse
from oxford_pet import *
from models.unet import *
#from models.resnet34_unet import *
from utils import *
from evaluate import *

def train(model,args,dataloader,num_batch,device):
    # implement the training function here
    if device=="cuda":
        model=model.to(device)
        img_dataset,mask_dataset=img_dataset.to(device),mask_dataset.to(device)
    epoch=args.epochs
    batch_size=args.batch_size
    lr=args.learning_rate
    optim=torch.optim.Adam(model.parameters(), lr=lr)
    loss_fc=nn.BCEWithLogitsLoss()
    model.train()
    msg_span=10
    for i in range(epoch):
        print(f"epoch{i}:")
        total_loss=0
        total_score=0
        total_correct_pixel=0
        total_num_pixel=0
        batch_cnt=0
        for batch in dataloader:
            x = batch["image"]
            y = batch["mask"]

            pred=model(x)
            loss=loss_fc(pred, y)
            pred=torch.sigmoid(pred)
            score=dice_score(pred,y)
            pred=(pred>0.5).to(torch.float)
            correct_pixel=(pred==y).sum()
            num_pixel=torch.numel(y)

            total_loss+=loss.item()
            total_score+=score
            total_correct_pixel+=correct_pixel
            total_num_pixel+=num_pixel
            batch_cnt+=1

            loss.backward()
            optim.step()
            if batch_cnt%msg_span==0:
                print(f"{batch_cnt}th batch | loss:{loss.item():.8f} | dice score:{score:.8f} | accuracy:{correct_pixel/num_pixel:.8f} {correct_pixel/num_pixel*100:.2f}%")
        print(f"{i}th epoch | loss:{total_loss/num_batch:.8f} | dice score:{total_score/num_batch:.8f} | accuracy:{total_correct_pixel/total_num_pixel*100:.2f}%")
    show_image(x[0])
    show_mask(pred[0])
    show_mask(y[0])

    return
    assert False, "Not implemented yet!"

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    #& C:/Users/darren/miniconda3/envs/pytest/python.exe c:/Users/darren/Documents/大學作業/深度/lab3/Lab3_Binary_Semantic_Segmentation/src/train.py -e 3 -b 16 -lr 0.0001
    #將doubleconv變成cbncbn，增加msg_span，等等再試試
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    use_cuda=torch.cuda.is_available()
    device=torch.device("cpu")
    if use_cuda:
        device=torch.device("cuda")
    print(f"use cuda:{use_cuda}")
    model=unet(in_channel=3,out_channel=1)
    dataloader,num_batch=load_dataset(data_path="dataset", mode="train", batchsize=args.batch_size)
    train(model,args,dataloader=dataloader,num_batch=num_batch,device=device)
    file_name=f"saved_models/unet_{args.epochs}.pt"
    save(model=model,file_name=file_name)
    print("test...")
    load(model=model,file_name=file_name)
    dataloader,num_batch=load_dataset(data_path="dataset", mode="valid", batchsize=args.batch_size)
    evaluate(model,dataloader=dataloader,num_batch=num_batch,device=device)

    