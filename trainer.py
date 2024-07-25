# implement your training script here
from Dataloader import *
from model.SCCNet import *
from utils import *
import numpy as np

modelist=["SD","LOSO","LOSO-FT"]
mode=modelist[0]

todolist=["train","test"]
todo=todolist[0]

if __name__=="__main__":
    lr=0.2
    epochs=100
    batch_size=288
    train_data=MIBCI2aDataset("train",mode,batch_size)
    test_data=MIBCI2aDataset("test",mode)
    ft_data=MIBCI2aDataset("finetune",mode)
    model=SCCNet(numClasses=train_data.features[0].shape[0],C=train_data.features[0].shape[2],timeSample=train_data.features[0].shape[3],Nu=22,Nc=20,Nt=1,dropoutRate=0.5)
    optimizer=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,gamma=0.99)
    loss_func=nn.CrossEntropyLoss()
    if todo=="test":
        model.load_state_dict(torch.load(f"model_weight_{mode}.pt"))
        model.eval()
        model.forward(test_data.features)
        loss=loss_func(model.o,test_data.labels)
        accuracy=torch.eq(model.y,test_data.labels).sum().item()/test_data.labels.shape[0]
        print(f"test | loss:{loss.item()} accuracy:{accuracy}")
    else:
        show_span=10
        accuracy_history=[]
        accuracy_history2=[]

        #train
        for i in range(epochs):
            for j in range(train_data.batch_num):
                optimizer.zero_grad()
                model.train()
                model.forward(train_data.features[j])
                loss=loss_func(model.o,train_data.labels[j])
                accuracy=torch.eq(model.y,train_data.labels[j]).sum().item()/train_data.labels[j].shape[0]
                loss.backward()
                optimizer.step()
                if i%show_span==0 and j==0:
                    print(f"1st batch eps:{i} loss:{loss.item()} accuracy:{accuracy}")
                    accuracy_history.append(loss.item())

                    model.eval()
                    model.forward(test_data.features)
                    loss=loss_func(model.o,test_data.labels)
                    accuracy=torch.eq(model.y,test_data.labels).sum().item()/test_data.labels.shape[0]
                    accuracy_history2.append(loss.item())

            scheduler.step()
        #draw
        draw(accuracy_history,epochs,show_span,"Loss")
        train_test_plot(accuracy_history,accuracy_history2,epochs,show_span,"Loss")
        #finetune
        if mode=="LOSO-FT":
            print("===finetune===")
            ftep=100
            cnt=0
            optimizer=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
            for param in model.parameters():
                if cnt==6:
                    break
                param.requires_grad = False
                cnt+=1
            accuracy_history=[]
            model.train()
            for i in range(ftep):
                optimizer.zero_grad()
                model.forward(ft_data.features)
                loss=loss_func(model.o,ft_data.labels)
                accuracy=torch.eq(model.y,ft_data.labels).sum().item()/ft_data.labels.shape[0]
                if i%show_span==0:
                    print(f"eps:{i} loss:{loss.item()} accuracy:{accuracy}")
                    accuracy_history.append(accuracy)
                loss.backward()
                optimizer.step()
            draw(accuracy_history,ftep,show_span,"Loss")
            print("===end===")
        #test
        model.eval()
        model.forward(test_data.features)
        loss=loss_func(model.o,test_data.labels)
        accuracy=torch.eq(model.y,test_data.labels).sum().item()/test_data.labels.shape[0]
        print(f"test | loss:{loss.item()} accuracy:{accuracy}")
        #save modelist=["SD","LOSO","LOSO-FT"]
        if mode=="SD" and accuracy>0.6:
            torch.save(model.state_dict(),f"model_weight_{mode}.pt")
        if mode=="LOSO" and accuracy>0.5:
            torch.save(model.state_dict(),f"model_weight_{mode}.pt")
        if mode=="LOSO-FT" and accuracy>0.7:
            torch.save(model.state_dict(),f"model_weight_{mode}.pt")
    ###
