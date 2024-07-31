from utils import *
import torch.nn as nn
def evaluate(model,dataloader,num_batch,device):
    # implement the evaluation function here
    with torch.no_grad():
        loss_fc=nn.BCEWithLogitsLoss()
        model.eval()
        total_loss=0
        total_score=0
        total_accuracy=0
        batch_cnt=0
        for batch in dataloader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)

            pred=model(x)
            loss=loss_fc(pred, y)
            pred=torch.sigmoid(pred)
            pred=(pred>0.5).to(torch.float)
            score=dice_score(pred,y)
            correct_pixel=(pred==y).sum()
            num_pixel=torch.numel(y)
            accuracy=correct_pixel/num_pixel

            total_loss+=loss.item()
            total_score+=score
            total_accuracy+=accuracy
            batch_cnt+=1
            #print(f"t | {batch_cnt}th batch | loss:{loss.item():.8f} | dice score:{score:.8f} | accuracy:{correct_pixel}/{num_pixel} {accuracy*100:.2f}%")
        print(f"test | loss:{total_loss/num_batch:.8f} | dice score:{total_score/num_batch:.8f} | accuracy:{total_accuracy/num_batch*100:.2f}%")
        show_image(x[0])
        show_mask(pred[0])
        show_mask(y[0])
    return
    assert False, "Not implemented yet!"