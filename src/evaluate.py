from utils import *
import torch.nn as nn
def evaluate(model,dataloader,num_batch,device):
    # implement the evaluation function here
    if device=="cuda":
        model=model.to(device)
        img_dataset,mask_dataset=img_dataset.to(device),mask_dataset.to(device)
    loss_fc=nn.BCEWithLogitsLoss()
    model.eval()
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

        print(f"{batch_cnt}th batch | loss:{loss.item():.8f} | dice score:{score:.8f} | accuracy:{correct_pixel/num_pixel:.8f} {correct_pixel/num_pixel*100:.2f}%")
    print(f"test | loss:{total_loss/num_batch:.8f} | dice score:{total_score/num_batch:.8f} | accuracy:{total_correct_pixel/total_num_pixel*100:.2f}%")
    show_image(x[0])
    show_mask(pred[0])
    show_mask(y[0])
    return
    assert False, "Not implemented yet!"