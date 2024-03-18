import os
import time
from test import test_single_model
import numpy as np
import torch
import wandb
from models import *


def train_ensemble(cfgs, opt, junior_index):
    out_dir = cfgs["Exp"]["out_dir"]

    Model = cfgs["Model"]
    network = Model["network"]
    mp = Model["mp"]
    ls = Model["ls"]
    mem_dim = Model["mem_dim"]
    shrink_thres = Model["shrink_thres"]
    layer = Model["layer"]

    Data = cfgs["Data"]
    dataset = Data["dataset"]
    img_size = Data["img_size"]

    Solver = cfgs["Solver"]
    bs = Solver["bs"]
    lr = Solver["lr"]
    weight_decay = Solver["weight_decay"]
    num_epoch = Solver["num_epoch"]
    
    train_loader = get_loader(
        dataset=dataset, dtype="train", bs=bs, img_size=img_size, workers=1
    )

    test_loader = get_loader(
        dataset=dataset, dtype="test", bs=1, img_size=img_size, workers=1
    )
    seniors = []

    for i in range(junior_index):
        senior = get_model(
            network=network,
            mp=mp,
            ls=ls,
            img_size=img_size,
            mem_dim=mem_dim,
            shrink_thres=shrink_thres,
            layer = layer
        )
        model_path = os.path.join(out_dir, "{}".format(opt.mode))
        senior.load_state_dict(
            torch.load(
                os.path.join(model_path, "{}.pth".format(i)),
                map_location=torch.device("cuda:{}".format(opt.gpu)),
            )
        )
        seniors.append(senior)
        print("load trained senior, ", i)

    junior = get_model(network=network, mp=mp, ls=ls, img_size=img_size, layer=layer)

    optimizer = torch.optim.Adam(
        junior.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay
    )

    logger = wandb.init(
        project="D2UE",
        config={"data": dataset, "mode": opt.mode, "network": network},
        name=f"{dataset}_{opt.mode}",
    )

    if network in ["AE", "AE-U"]:
        model = AE_trainer(
            junior,
            seniors,
            train_loader,
            test_loader,
            optimizer,
            num_epoch,
            logger,
            cfgs,
            opt,
        )
    elif network == "MemAE":
        model = MemAE_trainer(
            junior,
            seniors,
            train_loader,
            test_loader,
            optimizer,
            num_epoch,
            logger,
            cfgs,
            opt,
        )
    logger.finish()

    model_path = os.path.join(out_dir, "{}".format(opt.mode))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_name = os.path.join(model_path, "{}.pth".format(junior_index))
    torch.save(model.state_dict(), model_name)


def AE_trainer(
    junior, seniors, train_loader, test_loader, optimizer, num_epoch, logger, cfgs, opt
):
    t0 = time.time()
    [model.eval() for model in seniors]

    for e in range(1, num_epoch + 1):
        l1s, l2s, rars = [], [], []
        junior.train()
        for (x, _, _) in train_loader:
            x = x.cuda()
            x.requires_grad = False
            if cfgs["Model"]["network"] == "AE":

                out = junior(x)
                x_hat, cur_feature = out["x_hat"], out["features"]
                rec_err = (x_hat - x) ** 2

                features = [model(x)["features"] for model in seniors]

                RAR_Loss = [CKA(cur_feature, feature.detach()) for feature in features]

                if RAR_Loss == []:
                    RAR_Loss = torch.tensor([0], dtype=float, device="cuda")
                else:
                    RAR_Loss = torch.mean(torch.stack(RAR_Loss), dim=0)

                loss = rec_err.mean() + RAR_Loss
                l1s.append(rec_err.mean().item())
                rars.append(RAR_Loss.item())

            else:  # AE-U
                out = junior(x)
                mean, logvar, cur_feature = (
                    out["x_hat"],
                    out["log_var"],
                    out["features"],
                )
                rec_err = (mean - x) ** 2
                loss1 = torch.mean(torch.exp(-logvar) * rec_err)
                loss2 = torch.mean(logvar)

                features = [model(x)["features"] for model in seniors]
                RAR_Loss = [CKA(cur_feature, feature.detach()) for feature in features]

                if RAR_Loss == []:
                    RAR_Loss = torch.tensor([0], dtype=float, device="cuda")
                else:
                    RAR_Loss = torch.mean(torch.stack(RAR_Loss), dim=0)

                loss = loss1 + loss2 + cfgs["Solver"]["Lambda"] * RAR_Loss

                l1s.append(rec_err.mean().item())
                l2s.append(loss2.item())
                rars.append(RAR_Loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        l1s = np.mean(l1s)
        l2s = np.mean(l2s) if len(l2s) > 0 else 0
        rars = np.mean(rars)

        logger.log(
            step=e,
            data={
                "train/rec_loss": round(float(l1s), 4),
                "train/log_loss": round(float(l2s), 4),
                "train/RAR_Loss": round(float(rars), 4),
            },
        )

        if e % 25 == 0 or e == 1:
            t = time.time() - t0
            t0 = time.time()
            auc, ap = test_single_model(
                model=junior, test_loader=test_loader, cfgs=cfgs
            )
            logger.log(
                step=e,
                data={
                    "train/AUC": round(float(auc), 4),
                    "train/AP": round(float(ap), 4),
                },
            )

            print(
                "Mode {}. Epoch[{:3d}/{}]  Time:{:.2f}s  AUC:{:.3f}  AP:{:.3f}   "
                "Rec_err:{:.4f}  RAR_Loss:{:.4f} logvars:{:.4f}".format(
                    opt.mode, e, num_epoch, t, auc, ap, l1s, rars, l2s
                )
            )

    return junior


def MemAE_trainer(
    junior, seniors, train_loader, test_loader, optimizer, num_epoch, logger, cfgs, opt
):
    criterion_entropy = EntropyLossEncap()
    entropy_loss_weight = cfgs["Model"]["entropy_loss_weight"]
    [model.eval() for model in seniors]

    t0 = time.time()
    for e in range(1, num_epoch + 1):

        l1s, ent_ls, rars = [], [], []
        junior.train()
        for (x, _, _) in train_loader:
            x = x.cuda()
            x.requires_grad = False
            out = junior(x)

            x_hat, cur_feature, att_w = out["x_hat"], out["features"], out["att"]
            features = [model(x)["features"] for model in seniors]
            RAR_Loss = [CKA(cur_feature, feature.detach()) for feature in features]

            if RAR_Loss == []:
                RAR_Loss = torch.tensor([0], dtype=float, device="cuda")
            else:
                RAR_Loss = torch.mean(torch.stack(RAR_Loss), dim=0)

            rec_err = (x_hat - x) ** 2
            loss1 = rec_err.mean()
            entropy_loss = criterion_entropy(att_w)

            loss = loss1 + entropy_loss_weight * entropy_loss + cfgs["Solver"]["Lambda"] * RAR_Loss

            l1s.append(rec_err.mean().item())
            ent_ls.append(entropy_loss.mean().item())
            rars.append(RAR_Loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        l1s = np.mean(l1s)
        ent_ls = np.mean(ent_ls)
        rars = np.mean(rars)

        logger.log(
            step=e,
            data={
                "train/rec_loss": round(float(l1s), 4),
                "train/entropy_loss": round(float(ent_ls), 4),
                "train/RAR_Loss": round(float(rars), 4),
            },
        )

        if e % 25 == 0 or e == 1:
            t = time.time() - t0
            t0 = time.time()
            auc, ap = test_single_model(
                model=junior, test_loader=test_loader, cfgs=cfgs
            )
            logger.log(
                step=e,
                data={
                    "train/AUC": round(float(auc), 4),
                    "train/AP": round(float(ap), 4),
                },
            )
            print(
                "Mode {}. Epoch[{:3d}/{}]  Time:{:.2f}s  AUC:{:.3f}  AP:{:.3f}   "
                "Rec_err:{:.4f}   Entropy_loss:{:.4f}".format(
                    opt.mode, e, num_epoch, t, auc, ap, l1s, ent_ls
                )
            )

    return junior

