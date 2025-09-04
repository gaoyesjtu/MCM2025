import torch
import pandas as pd
from sympy.abc import alpha
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split

#--------
#1 read data,define dataframe
#-------

df=pd.read_csv(r"C:\Users\47797\Desktop\data.train.csv")
X=df.drop(columns=[df.columns[-1],df.columns[0]])#delete id and aim
y=df['tested_positive.4']

#--------
#2 change into torch tensor
#-------

X_tensor=torch.tensor(X.values,dtype=torch.float)
y_tensor=torch.tensor(y.values,dtype=torch.float)

#---------
#3 define Dataset and Dataloader
#--------

class myDataset(Dataset):
    def __init__(self, X, y):
        self.X=torch.tensor(X.values,dtype=torch.float)
        self.y=torch.tensor(y.values,dtype=torch.float)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#---------
#4 divide train and val
#---------
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42)

y_mean=y_train.mean()
y_std=y_train.std()
y_train=(y_train-y_mean)/y_std
y_val=(y_val-y_mean)/y_std

ds_train=myDataset(X_train,y_train)
ds_val=myDataset(X_val,y_val)
#--------
#5 dataloader
#--------

train_loader = DataLoader(ds_train,batch_size=32,shuffle=True)
val_loader = DataLoader(ds_val,batch_size=128,shuffle=False)

#------
#6 manu-MLP
#------
#deminsion
D=X_train.shape[1]
K=1
#parameters
g=torch.Generator().manual_seed(42)
W1=torch.randn(D,64,generator=g)*(2.0/D)**0.5
b1=torch.zeros(64)
W2=torch.randn(64,32,generator=g)*(2.0/64)**0.5
b2=torch.zeros(32)
W3=torch.randn(32,K,generator=g)*(2.0/32)**0.5
b3=torch.zeros(K)
#grad_require
for i in (W1,b1,W2,b2,W3,b3):
    i.requires_grad_(True)

#MLP
def linear(W,b,X):
    return X@W+b
def relu(X):
    return X.clamp_min(0)

#-----
#7 loss function
#-----

def loss_mean(yb,yp):
    return (((yb-yp)**2).mean())**0.5

#-----
#8 optimizer
#-----
sg={i:torch.zeros_like(i) for i in (W1,b1,W2,b2,W3,b3)}
#-----
#training
#------
#epch
lr=0.001
alpha=0.9
for epoch in range(100):
    train_loss=0.0
    for x,yb in train_loader:
        z1=linear(W1,b1,x)
        a1=relu(z1)
        z2=linear(W2,b2,a1)
        a2=relu(z2)
        z3=linear(W3,b3,a2)
        yb = yb.unsqueeze(1).float()
        loss=loss_mean(yb,z3)
        #remove grad
        for i in (W1,b1,W2,b2,W3,b3):
            if i.grad is not None:
                i.grad.zero_()
        #auto-grading
        loss.backward()
        #close RMSProp
        with torch.no_grad():
            for i in (W1,b1,W2,b2,W3,b3):
                sg[i]=sg[i]*alpha+(1-alpha)*i.grad**2
                i.data=i.data-lr*i.grad/(sg[i]**0.5+1e-8)

        train_loss+=loss.item()
    avg_trainloss = train_loss / len(train_loader)
    print(avg_trainloss)

#----
#testing
#-----

test_loss=0.0
for x,yb in val_loader:
    z1 = linear(W1, b1, x)
    a1 = relu(z1)
    z2 = linear(W2, b2, a1)
    a2 = relu(z2)
    z3 = linear(W3, b3, a2)
    yb = yb.unsqueeze(1).float()
    loss = loss_mean(yb, z3)
    test_loss+=loss.item()
avg_testloss = test_loss / len(val_loader)
print("-----------")
print(avg_testloss)



