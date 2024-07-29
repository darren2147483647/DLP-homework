import argparse
from train import *

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    # & C:/Users/darren/miniconda3/envs/plzcuda/python.exe c:/Users/darren/Documents/大學作業/深度/lab3/Lab3_Binary_Semantic_Segmentation/src/inference.py --model "unet_endless_70.pt" -b 8
    # & C:/Users/darren/miniconda3/envs/plzcuda/python.exe c:/Users/darren/Documents/大學作業/深度/lab3/Lab3_Binary_Semantic_Segmentation/src/inference.py --model "resnet_endless_330.pt" -b 8
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    use_cuda=torch.cuda.is_available()
    device=torch.device("cpu")
    if use_cuda:
        device=torch.device("cuda")
    print(f"use cuda:{use_cuda}")
    mode="unet"#"unet" "resnet"
    if mode=="unet":
        model=unet(in_channel=3,out_channel=1).to(device)
    else:
        model=res_unet(in_channel=3,out_channel=1).to(device)
    
    file_name=f"saved_models/{args.model}"
    load(model=model,file_name=file_name)
    dataloader,num_batch=load_dataset(data_path="dataset/oxford-iiit-pet", mode="valid", batchsize=args.batch_size)
    evaluate(model,dataloader=dataloader,num_batch=num_batch,device=device)
    
    #assert False, "Not implemented yet!"