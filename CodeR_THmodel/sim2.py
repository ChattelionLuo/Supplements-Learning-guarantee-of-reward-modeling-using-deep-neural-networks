import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os


from funcs import *

torch.set_default_dtype(torch.float64)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--iloop', type=int, default=7)
line_args = parser.parse_args()
idx_data = line_args.iloop

from itertools import product

widths=64
depths=4
noise_level=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
list_sim=[i for i in range(50)]

list_allset = list(product(*[noise_level,list_sim]))
noise= list_allset[idx_data][0]
sim = list_allset[idx_data][1]

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("Done folder") 
    else:
        print("Folder Already")


mkdir("model")
mkdir("result")
mkdir("logs")

n = 2**11
nval = 2**10
d = 10

torch.manual_seed(2024)
np.random.seed(2024)

true_weights = torch.randn(d, 1)
comparison_model= 'thurstonian'
normal_dist = torch.distributions.Normal(0, 1)
batch_size = 128
epochs=200


print("Random Seed number for data")
print( 1000*sim)
torch.manual_seed( 1000*sim)
np.random.seed(  1000*sim)

data_train = generate_synthetic_dataset2(n, d,comparison_model,true_weights, 2025*sim, noise)
data_val = generate_synthetic_dataset2(nval, d, comparison_model,true_weights, 2026*sim, noise)
data_test = generate_synthetic_dataset(n, d, comparison_model,true_weights, 2027*sim)
x_test=data_test[:][0]
y_test=data_test[:][1]


THE_SIM_MARK = "The_widths_{}_depths_{}_band_{}_sim_{}_".format(widths,depths,noise,sim)


dim_vec=[d]+[widths]*depths+[1]

train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(dataset=data_val, batch_size=len(data_val), shuffle=False)
loss_function = nn.BCELoss()
best_loss = 1e5
trigger_times = 0
model = FNN(dim_vec=dim_vec)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-7)

for epoch in range(epochs):
    model = model.train()
    for step, (x, y) in enumerate(train_loader):
        x, y = Variable(x, requires_grad=True), Variable(y)
        output = model(x)
        output = output.squeeze()

        #loss =loss_function(torch.sigmoid(output),y[:,0])  #BT
        loss =loss_function(normal_dist.cdf(output),y[:,0]) #thurstonian
        lossTrain = loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss = 0
    model = model.eval()
    for x_val,y_val in eval_loader:
        output_val = model(x_val)
        output_val = output_val.squeeze()

        #lossii = loss_function(torch.sigmoid(output_val), y_val[:,0]) #BT
        lossii = loss_function(normal_dist.cdf(output_val), y_val[:,0]) #thurstonian
        
        loss = loss +  lossii
    current_loss = loss / len(eval_loader)
    if epoch % 20==0:
        print("epoch:{},Tloss:{}, loss:{}".format(epoch,lossTrain, current_loss))
    if current_loss < best_loss:
        best_loss = current_loss
        trigger_times = 0
        torch.save(model.state_dict(), os.path.join("./model/", "dCOR_idata_" + THE_SIM_MARK + '_net.pt'))
    else:
        trigger_times += 1
        if trigger_times >= 20: #patience
            print('Early stopping!')
            break

###read the best model
file_net_dict =  os.path.join("./model/", "dCOR_idata_" + THE_SIM_MARK + '_net.pt')
model.load_state_dict(torch.load(file_net_dict))

pred_test = model(x_test).detach()

x_test2  = x_test
for k in range(d):
    x_test2[:,k] = torch.sin(x_test2[:,k])
reg_test=torch.mean((( 4*torch.sin(4* x_test2 @ true_weights)>0) != (pred_test >0)) * torch.abs(4 *torch.sin(4*x_test2 @ true_weights))) 

dict_res = {#"pred_test":pred_test,
            "reg_test":reg_test}
import pickle
with open(os.path.join("./result/",  "dCOR_idata_" + THE_SIM_MARK + '_res.pickle'), 'wb') as handle:
    pickle.dump(dict_res, handle, protocol=pickle.HIGHEST_PROTOCOL)