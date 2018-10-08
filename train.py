import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm_notebook as tqdm

def loss_batch(model, xb, yb, loss_fn, metric=None, opt=None):
    "Calculate loss for the batch `xb,yb` and backprop with `opt`"
    xb = xb.permute(0,3,1,2)
    yb_ = model(xb)
    loss = loss_fn(yb_, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    return float(loss), len(yb), None if metric is None else metric(yb_, yb).item()

def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, metric=None) -> None:
    "Train `model` for `epochs` with `loss_fun` and `optim`"
    for epoch in range(epochs):
        
        tq = tqdm(total=(len(train_dl) * train_dl.batch_size))
        tq.set_description('Epoch {}'.format(epoch))
        
        #model.train()
        for xb,yb in train_dl:
            xb = torch.autograd.Variable(xb).cuda()
            loss, yb_len, _ = loss_batch(model, xb, yb.cuda(), loss_fn, opt=opt, metric=metric)
            tq.update(yb_len)
            
            tq.set_postfix(loss='{:.5f}'.format(loss))

        model.eval()
        with torch.no_grad():
            losses,nums = zip(*[loss_batch(model, xb, yb, loss_fn)
                                for xb,yb in valid_dl])
        val_loss = np.sum(np.multiply(losses,nums)) / np.sum(nums)
        print(f'Val Loss {val_loss}')

    tq.close()