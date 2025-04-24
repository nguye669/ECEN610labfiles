import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import torch_directml                           
    DEVICE  = torch_directml.device()
    BACKEND = "DirectML"
except ModuleNotFoundError:
    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BACKEND = "CUDA/HIP" if DEVICE.type == "cuda" else "CPU"
VDD = 1.2
A_H, TAU   = 0.68, 1.45
V_OUT_MAX  = 1.1
K_SOFT     = 25.0

def f_nmos(v): return 2.5 * torch.clamp(v - 0.20, min=0.)**2
def g_pmos(i): return torch.clamp(A_H * (1. - torch.exp(-TAU * i)), 0., VDD)
def soft_cs(v): return V_OUT_MAX * torch.sigmoid(K_SOFT * v)
q4 = lambda w: torch.clamp(torch.round(w), -8, 7) + (w - w.detach())
class HiddenLayer(nn.Module):
    def __init__(self, in_feats=25, out_feats=28):
        super().__init__()
        self.w = nn.Parameter(torch.empty(out_feats, in_feats))
        nn.init.uniform_(self.w, -4, 4)
    def forward(self, x):
        wq = q4(self.w)
        i  = f_nmos(x.unsqueeze(1))
        i_p = (torch.clamp(+wq, min=0).unsqueeze(0) * i).sum(-1)
        i_n = (torch.clamp(-wq, min=0).unsqueeze(0) * i).sum(-1)
        return g_pmos(i_p) - g_pmos(i_n)

class OutputLayer(nn.Module):
    def __init__(self, in_feats=28, classes=10):
        super().__init__()
        self.w = nn.Parameter(torch.empty(classes, in_feats))
        nn.init.uniform_(self.w, -4, 4)
    def forward(self, h):
        vdiff = (h.unsqueeze(1) * q4(self.w).unsqueeze(0)).sum(-1)
        return soft_cs(vdiff)

class AnalogANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid, self.out = HiddenLayer(), OutputLayer()
    def forward(self, x):
        v = self.out(self.hid(x))                 
        return v / v.sum(dim=1, keepdim=True)     
def to_voltage(t): return 0.30 + (t - 0.5) * 0.30
class Flatten:  
    def __call__(self, t): return t.view(-1)

to5x5 = transforms.Compose([
    transforms.Resize((5,5), transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Lambda(to_voltage),
    Flatten(),
])

def loaders(batch=128, workers=12):
    root = Path.home() / 'mnist'
    tr = datasets.MNIST(root, True,  download=True, transform=to5x5)
    te = datasets.MNIST(root, False, download=True, transform=to5x5)
    return (DataLoader(tr, batch, shuffle=True,  num_workers=workers, pin_memory=True),
            DataLoader(te,  256,  shuffle=False, num_workers=workers, pin_memory=True))
def train_and_evaluate(epochs=10):
    tr_ld, te_ld = loaders()
    net   = AnalogANN().to(DEVICE)
    opt   = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.99,nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.5,steps_per_epoch=len(tr_ld), epochs=epochs, pct_start=0.3)
    mse_hist = []

    for ep in range(epochs):
        net.train(); running = n = 0
        for x, y in tr_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            probs = net(x)
            classes = torch.arange(10, device=DEVICE)
            target  = (y.unsqueeze(1) == classes).float()
            loss = F.mse_loss(probs, target)
            loss.backward(); opt.step(); sched.step()
            running += loss.item() * y.size(0); n += y.size(0)
        mse = running / n; mse_hist.append(mse)
        print(f"Epoch {ep:02d}  MSE = {mse:.4f}")
    net.eval(); correct = 0
    conf = torch.zeros(10,10,dtype=torch.int32)
    with torch.no_grad():
        for x, y in te_ld:
            preds = net(x.to(DEVICE)).argmax(1).cpu()
            correct += (preds==y).sum().item()
            for t,p in zip(y,preds): conf[p.item(), t.item()] += 1
    print(f"\nTest accuracy ≈ {100*correct/len(te_ld.dataset):.1f} %")
    return mse_hist, conf
def plot_results(mse_hist, conf):
    i = np.linspace(0,2,300); v = np.linspace(-1,1,300)
    plt.figure(figsize=(4,3))
    plt.plot(i, g_pmos(torch.from_numpy(i)).numpy()); plt.grid()
    plt.title('Hidden-layer g(I)'); plt.xlabel('Σ current'); plt.ylabel('V_out (V)')
    plt.figure(figsize=(4,3))
    plt.plot(v, soft_cs(torch.from_numpy(v)).numpy()); plt.grid()
    plt.title('Output-layer soft_cs'); plt.xlabel('V_P−V_N (V)'); plt.ylabel('V_out (V)')
    plt.figure(figsize=(5,4))
    plt.imshow(conf, cmap='Greys')
    for r in range(10):
        for c in range(10):
            val = int(conf[r,c])
            plt.text(c,r,val,ha='center',va='center',
                     color='white' if val > conf.max()*0.6 else 'black', fontsize=7)
    plt.title('Confusion Matrix'); plt.xlabel('Target Class'); plt.ylabel('Output Class')
    plt.xticks(range(10)); plt.yticks(range(10))
    plt.figure(figsize=(4,3))
    plt.plot(range(len(mse_hist)), mse_hist, marker='o')
    plt.title('Average MSE per Epoch'); plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.grid()
    plt.tight_layout(); plt.show()
if __name__ == '__main__':
    mse_history, conf_mat = train_and_evaluate(epochs=100)
    plot_results(mse_history, conf_mat)
