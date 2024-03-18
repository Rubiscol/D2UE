import yaml
from trainer import *
from test import evaluate_ens, evaluate_unc, evaluate_grad
from argparse import ArgumentParser

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str, default="config/RSNA_AE.yaml"
    )  
    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default=None,
        help="e.g., train, eval_ens, eval_unc, eval_grad",
    )
    parser.add_argument("--gpu", dest="gpu", type=int, default=1, help="gpu")
    parser.add_argument("--k", dest="k", type=int, default=3, help="Number of ensembles")

    opt = parser.parse_args()
    
    with open(opt.config, "r") as f:
        cfgs = yaml.safe_load(f)
    
    setup(cfgs, opt)
    
    if opt.mode == "train":
        for i in range(opt.k):
            print("=> Training the junior {}".format(i))
            train_ensemble(cfgs, opt, i)
            
    elif opt.mode == "eval_ens":
        print("=> Evaluating ensemble reconstruction ...")
        evaluate_ens(cfgs, opt)
        
    elif opt.mode == "eval_unc":
        print("=> Evaluating output uncertainty ...")
        evaluate_unc(cfgs, opt)

    elif opt.mode == "eval_grad":
        print("=> Evaluating dual-space uncertainty ...")
        evaluate_grad(cfgs, opt)

    else:
        raise Exception("Invalid mode: {}".format(opt.mode))
