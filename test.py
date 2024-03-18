import torch
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import os
from models import get_loader, get_model, load_models
import matplotlib.pyplot as plt
from torch.autograd import Variable


def chi2_distance(A, B):
    # compute the chi-squared distance
    chi = 0.5 * np.sum(
        [((a - b) ** 2 + 1e-15) / (a + b + 1e-15) for (a, b) in zip(A, B)]
    )

    return chi


def anomaly_score_histogram(y_score, y_true, anomaly_score, out_dir, f_name):
    plt.cla()

    normal_score, _, _ = plt.hist(
        y_score[y_true == 0],
        bins=100,
        range=(0, 1),
        density=True,
        color="blue",
        alpha=0.5,
        label="Normal",
        edgecolor="none",
    )
    abnormal_score, _, _ = plt.hist(
        y_score[y_true == 1],
        bins=100,
        range=(0, 1),
        density=True,
        color="red",
        alpha=0.5,
        label="Abnormal",
        edgecolor="none",
    )

    chi2_dis = chi2_distance(normal_score, abnormal_score)
    plt.text(0.5, 6, "$\chi^2$-dis={:.2f}".format(chi2_dis))

    plt.xlabel(anomaly_score)
    plt.ylabel("Frequency")
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0, 2, 4, 6, 8, 10])
    plt.xlim(0, 1)
    plt.ylim(0, 11)
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}/{}.png".format(out_dir, f_name), dpi=300)


def test_single_model(model, test_loader, cfgs):
    model.eval()
    network = cfgs["Model"]["network"]
    with torch.no_grad():
        y_score, y_true = [], []
        for bid, (x, label, img_id) in enumerate(test_loader):

            x = x.cuda()
            if network == "AE-U":
                out = model(x)
                out, logvar = out["x_hat"], out["log_var"]
                rec_err = (out - x) ** 2
                res = torch.exp(-logvar) * rec_err
            elif network == "AE":
                x_hat = model(x)["x_hat"]
                rec_err = (x_hat - x) ** 2
                res = rec_err
            elif network == "MemAE":
                recon_res = model(x)
                rec = recon_res["x_hat"]
                res = (rec - x) ** 2

            res = res.mean(dim=(1, 2, 3))

            y_true.append(label.cpu())
            y_score.append(res.cpu().view(-1))

        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        auc = metrics.roc_auc_score(y_true, y_score)
        ap = metrics.average_precision_score(y_true, y_score)
        return auc, ap


def evaluate_ens(cfgs, opt):
    Model = cfgs["Model"]
    network = Model["network"]
    Data = cfgs["Data"]
    dataset = Data["dataset"]
    img_size = Data["img_size"]
    out_dir = cfgs["Exp"]["out_dir"]

    test_loader = get_loader(
        dataset=dataset, dtype="test", bs=1, img_size=img_size, workers=1
    )

    models = load_models(cfgs, opt)
    print("=> Loaded {} models.".format(len(models)))

    print("=> Evaluating ... ")
    with torch.no_grad():
        y_true = []
        rec_err_l = []

        for x, label, img_id in tqdm(test_loader):
            x = x.cuda()
            if network == "AE":
                models_rec = torch.cat(
                    [model(x)["x_hat"].squeeze(0) for model in models]
                )
            elif network == "AE-U":

                models_rec, unc = [], []
                for model in models:
                    out = model(x)
                    mean, logvar = out["x_hat"], out["log_var"]
                    models_rec.append(mean.squeeze(0))
                    unc.append(torch.exp(logvar).squeeze(0))
                models_rec = torch.cat(models_rec)
                unc = torch.cat(unc)
            elif network == "MemAE":
                models_rec = torch.cat(
                    [model(x)["x_hat"].squeeze(0) for model in models]
                )
            else:
                raise Exception("Invalid Network")

            mu_b = torch.mean(models_rec, dim=0)  # h x w

            if network == "AE-U":
                var = torch.mean(unc, dim=0)
                rec_err = (x - mu_b) ** 2 / var
            else:
                rec_err = (x - mu_b) ** 2

            rec_err_l.append(rec_err.mean().cpu())
            y_true.append(label.cpu().item())

        rec_err_l = np.array(rec_err_l)
        y_true = np.array(y_true)

        rec_auc = metrics.roc_auc_score(y_true, rec_err_l)
        rec_ap = metrics.average_precision_score(y_true, rec_err_l)

        rec_str = "Rec. (ensemble)  AUC:{:.3f}  AP:{:.3f}".format(rec_auc, rec_ap)

        print(rec_str)

        with open(os.path.join(out_dir, "results.txt"), "w") as f:
            f.write(rec_str + "\n")
        print()

        rec_err_l = (rec_err_l - np.min(rec_err_l)) / (
            np.max(rec_err_l) - np.min(rec_err_l)
        )

        anomaly_score_histogram(
            y_score=rec_err_l,
            y_true=y_true,
            anomaly_score="Ensemble econstruction",
            out_dir=out_dir,
            f_name="rec_hist",
        )


def evaluate_unc(cfgs, opt):
    Model = cfgs["Model"]
    network = Model["network"]
    Data = cfgs["Data"]
    dataset = Data["dataset"]
    img_size = Data["img_size"]
    out_dir = cfgs["Exp"]["out_dir"]

    test_loader = get_loader(
        dataset=dataset, dtype="test", bs=1, img_size=img_size, workers=1
    )

    models = load_models(cfgs, opt)
    print("=> Loaded {} models.".format(len(models)))

    print("=> Evaluating ... ")
    with torch.no_grad():
        y_true = []
        unc_dis_l = []

        for x, label, img_id in tqdm(test_loader):
            x = x.cuda()
            if network == "AE":
                models_rec = torch.cat(
                    [model(x)["x_hat"].squeeze(0) for model in models]
                )
            elif network == "AE-U":
                models_rec, unc = [], []
                for model in models:
                    out = model(x)
                    mean, logvar = out["x_hat"], out["log_var"]
                    models_rec.append(mean.squeeze(0))
                    unc.append(torch.exp(logvar).squeeze(0))
                models_rec = torch.cat(models_rec)
                unc = torch.cat(unc)
            elif network == "MemAE":
                models_rec = torch.cat(
                    [model(x)["x_hat"].squeeze(0) for model in models]
                )
            else:
                raise Exception("Invalid Network")

            # Image-Level discrepancy
            if network == "AE-U":
                var = torch.mean(unc, dim=0)
                unc_dis = torch.sqrt(torch.var(models_rec, dim=0) / var)
            else:
                unc_dis = torch.std(models_rec, dim=0)

            unc_dis_l.append(unc_dis.mean().cpu())
            y_true.append(label.cpu().item())

        unc_dis_l = np.array(unc_dis_l)
        y_true = np.array(y_true)

        unc_auc = metrics.roc_auc_score(y_true, unc_dis_l)
        unc_ap = metrics.average_precision_score(y_true, unc_dis_l)

        unc_str = "Ensemble-unc       AUC:{:.3f}  AP:{:.3f}".format(unc_auc, unc_ap)

        print(unc_str)

        with open(os.path.join(out_dir, "results.txt"), "w") as f:
            f.write(unc_str + "\n")
        print()

        unc_dis_l = (unc_dis_l - np.min(unc_dis_l)) / (
            np.max(unc_dis_l) - np.min(unc_dis_l)
        )

        anomaly_score_histogram(
            y_score=unc_dis_l,
            y_true=y_true,
            anomaly_score="Ensemble uncertainty",
            out_dir=out_dir,
            f_name="unc_hist",
        )


def evaluate_grad(cfgs, opt):
    Model = cfgs["Model"]
    network = Model["network"]
    Data = cfgs["Data"]
    dataset = Data["dataset"]
    img_size = Data["img_size"]
    out_dir = cfgs["Exp"]["out_dir"]

    test_loader = get_loader(
        dataset=dataset, dtype="test", bs=1, img_size=img_size, workers=1
    )

    models = load_models(cfgs, opt)

    print("=> Loaded {} models".format(len(models)))

    print("=> Evaluating ... ")

    y_true = []
    unc_dis_l = []

    for x, label, img_id in tqdm(test_loader):

        x = Variable(x, requires_grad=True)
        x = x.cuda()

        if network == "AE" or network == "MemAE":

            grad_recs = []

            for model in models:

                rec = model(x)["x_hat"].squeeze(0)
                loss = torch.abs((rec - x))
                gradient = torch.autograd.grad(torch.mean(loss), x)[0].squeeze(0)
                grad_rec = loss * gradient
                grad_recs.append(grad_rec)

            grad_recs = torch.cat(grad_recs)

        elif network == "AE-U":
            grad_recs, unc = [], []
            for model in models:
                out = model(x)

                mean, logvar = out["x_hat"], out["log_var"]
                rec_err = (x - mean) ** 2
                loss = torch.mean(torch.exp(-logvar) * rec_err)
                gradient = torch.autograd.grad(torch.mean(loss), x)[0].squeeze(0)
                grad_rec = torch.abs(mean - x) * gradient

                grad_recs.append(grad_rec)
                unc.append(torch.exp(logvar).squeeze(0))

            grad_recs = torch.cat(grad_recs)
            unc = torch.cat(unc)

        else:
            raise Exception("Invalid Network")

        # Image-Level discrepancy
        if network == "AE-U":
            var = torch.mean(unc, dim=0)
            unc_dis = torch.std(grad_recs / torch.sqrt(var), dim=0)

        else:
            unc_dis = torch.std(grad_recs, dim=0)

        unc_dis_l.append(unc_dis.detach().mean().cpu())

        y_true.append(label.cpu().item())

    unc_dis_l = np.array(unc_dis_l)
    y_true = np.array(y_true)

    unc_auc = metrics.roc_auc_score(y_true, unc_dis_l)
    unc_ap = metrics.average_precision_score(y_true, unc_dis_l)

    unc_str = "Dual-space Uncertainty      AUC:{:.3f}  AP:{:.3f}".format(
        unc_auc, unc_ap
    )

    print(unc_str)

    with open(os.path.join(out_dir, "results.txt"), "w") as f:
        f.write(unc_str + "\n")

    print()

    unc_dis_l = (unc_dis_l - np.min(unc_dis_l)) / (
        np.max(unc_dis_l) - np.min(unc_dis_l)
    )
    anomaly_score_histogram(
        y_score=unc_dis_l,
        y_true=y_true,
        anomaly_score="D2UE",
        out_dir=out_dir,
        f_name="D2UE",
    )
