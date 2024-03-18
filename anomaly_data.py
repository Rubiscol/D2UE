import os
import time
from PIL import Image
from torch.utils import data
import json
from joblib import Parallel, delayed


def parallel_load(img_dir, img_list, img_size, verbose=0):
    return Parallel(n_jobs=-1, verbose=verbose)(delayed(
        lambda file: Image.open(os.path.join(img_dir, file)).convert("L").resize(
            (img_size, img_size), resample=Image.BILINEAR))(file) for file in img_list)


class AnomalyDetectionDataset(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train"):
        super(AnomalyDetectionDataset, self).__init__()
        assert mode in ["train", "test"]
        self.root = main_path
        self.labels = []
        self.img_id = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x

        with open(os.path.join(main_path, "data.json")) as f:
            data_dict = json.load(f)

        print("Loading images")
        if mode == "train":
            train_l = data_dict["train"]["0"]
            
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), train_l, img_size)
            self.labels += (len(train_l) ) * [0]
            self.img_id += [img_name.split('.')[0] for img_name in train_l]
            print("Loaded {} normal images, {:.3f}s.".format(len(train_l), time.time() - t0))

        else:  
            test_normal = data_dict["test"]["0"]
            test_abnormal = data_dict["test"]["1"]

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), test_l, img_size)
            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_id += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        img_id = self.img_id[index]
        return img, label, img_id

    def __len__(self):
        return len(self.slices)
