import torch
import random
from models import AE, AEU, MemAE
from anomaly_data import AnomalyDetectionDataset
from torchvision import transforms
from torch.utils import data
import os
import numpy as np
import torch
import numpy as np
import os


def get_model(network, in_channels=None, out_channels=None, mp=None, ls=None, img_size=None, mem_dim=None,
              shrink_thres=0.0, layer=4):
    if network == "AE":
        model = AE(latent_size=ls, expansion=mp, input_size=img_size, layer=layer)
    elif network == "AE-U":
        model = AEU(latent_size=ls, expansion=mp, input_size=img_size, layer=layer)
    elif network == "MemAE":
        model = MemAE(latent_size=ls, expansion=mp, input_size=img_size, layer=layer)
    else:
        raise Exception("Invalid Model Name!")

    model.cuda()
    return model

def setup(cfgs, opt):
    torch.cuda.set_device(opt.gpu)
    set_seed(cfgs["Solver"]["seed"])
    out_dir = cfgs["Exp"]["out_dir"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed)  
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_loader(dataset, dtype, bs, img_size, workers=1):

    DATA_PATH = os.path.join("./Med-AD")

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    print("Dataset: {}".format(dataset))
    if dataset == 'rsna':
        path = os.path.join(DATA_PATH, 'RSNA')
    elif dataset == 'vin':
        path = os.path.join(DATA_PATH, "VinCXR")
    elif dataset == 'brain':
        path = os.path.join(DATA_PATH, "BrainTumor")
    elif dataset == 'lag':
        path = os.path.join(DATA_PATH, "LAG")
    else:
        raise Exception("Invalid dataset: {}".format(dataset))

  
    dset = AnomalyDetectionDataset(main_path=path, transform=transform, mode=dtype, img_size=img_size)

    train_flag = True if dtype == 'train' else False
    dataloader = data.DataLoader(dset, bs, shuffle=train_flag,
                                 drop_last=train_flag, num_workers=workers, pin_memory=True)

    return dataloader


def load_models(cfgs, requires_grad=False):
    gpu = cfgs["Exp"]["gpu"]
    Model = cfgs["Model"]
    network = Model["network"]
    mp = Model["mp"]
    ls = Model["ls"]
    mem_dim = Model["mem_dim"]
    shrink_thres = Model["shrink_thres"]

    Data = cfgs["Data"]
    img_size = Data["img_size"]

    out_dir = cfgs["Exp"]["out_dir"]
    out_dir = os.path.join(out_dir, "train")

    models = []
    for state_dict in sorted(os.listdir(os.path.join(out_dir)), key=lambda x: int(x.split(".")[0])):
        model = get_model(network=network, mp=mp, ls=ls, img_size=img_size, mem_dim=mem_dim, shrink_thres=shrink_thres)
        model.load_state_dict(torch.load(os.path.join(out_dir, state_dict),
                                         map_location=torch.device('cuda:{}'.format(gpu))))
        model.eval()
        if not requires_grad:
            for param in model.parameters():
                param.requires_grad = False
        models.append(model)

    return models
