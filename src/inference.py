import argparse
from train import *

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    mode="unet"
    if mode=="unet":
        model=unet(in_channel=3,out_channel=1)
    else:
        pass
    use_cuda=torch.cuda.is_available()
    device=torch.device("cpu")
    if use_cuda:
        device=torch.device("cuda")
    print(f"use cuda:{use_cuda}")
    file_name=f"saved_models/{args.model}"
    load(model=model,file_name=file_name)
    dataloader,num_batch=load_dataset(data_path="dataset", mode="test", batchsize=args.batch_size)
    evaluate(model,dataloader=dataloader,num_batch=num_batch,device=device)
    
    #assert False, "Not implemented yet!"