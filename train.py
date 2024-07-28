import argparse
from oxford_pet import *
from models.unet import *
from models.resnet34_unet import *
from utils import *
from evaluate import *

def train(model,args,dataloader,num_batch,device,no_msg=False):
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
        if not no_msg:
            print(f"[epoch{i}:]")
        total_loss=0
        total_score=0
        total_accuracy=0
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
            accuracy=correct_pixel/num_pixel

            total_loss+=loss.item()
            total_score+=score
            total_accuracy+=accuracy
            batch_cnt+=1

            loss.backward()
            optim.step()
            if batch_cnt%msg_span==0 and not no_msg:
                print(f"{i} | {batch_cnt}th batch | loss:{loss.item():.8f} | dice score:{score:.8f} | accuracy:{correct_pixel}/{num_pixel} {accuracy*100:.2f}%")
        if not no_msg:
            print(f"[{i}th epoch | loss:{total_loss/num_batch:.8f} | dice score:{total_score/num_batch:.8f} | accuracy:{total_accuracy/num_batch*100:.2f}%]")
    if not no_msg:
        show_image(x[0])
        show_mask(pred[0])
        show_mask(y[0])
    else:
        print(f"score:{score} accuracy:{accuracy}")
    return
    assert False, "Not implemented yet!"

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    #& C:/Users/darren/miniconda3/envs/pytest/python.exe c:/Users/darren/Documents/大學作業/深度/lab3/Lab3_Binary_Semantic_Segmentation/src/train.py -e 5 -b 32 -lr 0.01
    #將doubleconv變成cbncbn，增加msg_span，等等再試試
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    #device
    use_cuda=torch.cuda.is_available()
    device=torch.device("cpu")
    if use_cuda:
        device=torch.device("cuda")
    print(f"use cuda:{use_cuda}")
    #choose model
    mode="unet"#"unet" "resnet"
    if mode=="unet":
        model=unet(in_channel=3,out_channel=1)
    else:
        model=res_unet(in_channel=3,out_channel=1)
    #retrain?
    retrain=True
    pretrain_epoch=5
    if retrain:
        file_name=f"saved_models/{mode}_{pretrain_epoch}.pt"
        load(model=model,file_name=file_name)
        accumulate_epoch=pretrain_epoch+args.epochs
    else:
        accumulate_epoch=args.epochs
    #endless mode?
    endless_mode=True
    ###
    dataloader,num_batch=load_dataset(data_path="dataset", mode="train", batchsize=args.batch_size)
    if endless_mode:
        for i in range(1000):
            file_name=f"saved_models/{mode}_endless_{accumulate_epoch}.pt"
            train(model,args,dataloader=dataloader,num_batch=num_batch,device=device,no_msg=True)
            save(model=model,file_name=file_name)
            accumulate_epoch+=args.epochs
    else:
        file_name=f"saved_models/{mode}_{accumulate_epoch}.pt"
        train(model,args,dataloader=dataloader,num_batch=num_batch,device=device)
        save(model=model,file_name=file_name)
        print("test...")
        load(model=model,file_name=file_name)
        dataloader,num_batch=load_dataset(data_path="dataset", mode="valid", batchsize=args.batch_size)
        evaluate(model,dataloader=dataloader,num_batch=num_batch,device=device)

    