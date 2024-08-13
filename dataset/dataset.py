import os
import re
import glob
import copy
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, FancyPCA, HueSaturationValue, \
    OneOf, ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur,Resize
from albumentations.pytorch.functional import img_to_tensor
import cv2
import logging
from glob import glob
from os import listdir
from os.path import join
import random

class ReadDataset():
    """
        initialize once and can be reused as train/test/validation set
    """

    def __init__(self, args):

        oversample = True
        self.data = {
            "train": [],
            "val":[],
            "test": [],
        }
        self.labels = copy.deepcopy(self.data)
        self.path = f'../_Datasets/{args.dataset}'
        self.root = self.path
        
        if 'celeb' in args.dataset:
            self.init_celeb_df()
        elif 'cifar' in args.dataset:
            self.init_cifar_10()
        elif 'waitan' in args.dataset:
            self.init_waitan()
        elif 'deepfake' in args.dataset:
            self.init_deepfake_sets()
        elif 'all' in args.dataset:
            self.init_all_datasets()
        
        info = f"train data: {len(self.data['train'])}, test data: {len(self.data['test'])}"
        logging.info(info)
        print(info)
    
    def init_all_datasets(self):
        real_list = glob('D:/_Datasets/adv/race-real/*') + glob('D:/_Datasets/deepfake/real/real-race-processed-9845/*')
        fake_list = glob('D:/_Datasets/adv/race-fake/*') + glob('D:/_Datasets/deepfake/fake/fake-race-9996/*')

        random.shuffle(real_list)
        random.shuffle(fake_list)
        if len(real_list) <= len(fake_list):
            fake_list = fake_list[0: len(real_list)]
        else:
            real_list = real_list[0: len(fake_list)]
        real_tgt_list = [0] * len(real_list) # !!! important real label = 1 in our race
        fake_tgt_list = [1] * len(fake_list) # !!! important fake label = 0 in our race
        
        # train = 0.8 * all, val == test = 0.2 * all
        ratio = 0.8
        train_real_idx = int(ratio*len(real_list))
        train_fake_idx = int(ratio*len(fake_list))
        self.data['train'] = real_list[0: train_real_idx] + fake_list[0: train_fake_idx]
        self.data['val'] = real_list[train_real_idx:] + fake_list[train_fake_idx:] 
        self.data['test'] = self.data['val']
                                      
        self.labels['train'] = real_tgt_list[0: train_real_idx] + fake_tgt_list[0: train_fake_idx]
        self.labels['val'] = real_tgt_list[train_real_idx:] + fake_tgt_list[train_fake_idx:]
        self.labels['test'] = self.labels['val']

    def init_deepfake_sets(self):
        real_f_list = glob('./datasets/real/*')
        fake_f_list = glob('./datasets/fake/*')
        
        real_list = []
        for item in real_f_list:
            _list = glob(item + '/*')
            if 'wanghong' in item:
                random.shuffle(_list)
                _list = _list[0: 20000]
            real_list.append(_list)
        real_list = [y for x in real_list for y in x] # about 42906

        fake_list = []
        for item in fake_f_list:
            _list = glob(item + '/*')
            fake_list.append(_list)
        fake_list = [y for x in fake_list for y in x] # about 53396

        random.shuffle(real_list)
        random.shuffle(fake_list)
        if len(real_list) <= len(fake_list):
            fake_list = fake_list[0: len(real_list)]
        else:
            real_list = real_list[0: len(fake_list)]
        real_tgt_list = [1] * len(real_list) # !!! important real label = 1 in our race
        fake_tgt_list = [0] * len(fake_list) # !!! important fake label = 0 in our race
        # print('total_real_imgs =', len(real_list))
        # print('total_fake_imgs =', len(fake_list))
        kaggle_train_data, kaggle_train_label_inv, kaggle_val_data, kaggle_val_label_inv = self.load_kaggle_datasets()
        dfdc_train_data, dfdc_train_label, dfdc_val_data, dfdc_val_label = self.load_dfdc_datasets()
        celeb_train_data, celeb_train_label, celeb_val_data, celeb_val_label = self.load_celeb_datasets()
        # ff_train_data, ff_train_label, ff_val_data, ff_val_label = self.load_ff_datasets()
        ratio = 0.8
        train_real_idx = int(ratio*len(real_list))
        train_fake_idx = int(ratio*len(fake_list))

        # train = 0.8 * all, val == test = 0.2 * all
        self.data['train'] = real_list[0: train_real_idx] + fake_list[0: train_fake_idx] + kaggle_train_data + \
            dfdc_train_data + celeb_train_data# + ff_train_data
        self.data['val'] = real_list[train_real_idx:] + fake_list[train_fake_idx:]  + kaggle_val_data + celeb_val_data + dfdc_val_data  # + ff_val_data 
            
        self.data['test'] = self.data['val']
                                      
        self.labels['train'] = real_tgt_list[0: train_real_idx] + fake_tgt_list[0: train_fake_idx] + kaggle_train_label_inv + \
            dfdc_train_label + celeb_train_label# + ff_train_label
        self.labels['val'] = real_tgt_list[train_real_idx:] + fake_tgt_list[train_fake_idx:]  + kaggle_val_label_inv + celeb_val_label + dfdc_val_label # + ff_val_label 
            
        self.labels['test'] = self.labels['val']
        print('train_len =', len(self.data['train']))
        print('test_len =', len(self.data['test']))
        # print(f'train_len = {len(self.data['train'])}')
        # print(f'test_len = {len(self.data['test'])}')

        # print(f'train_real_len = {len(real_list[0: train_real_idx])}')
        # print(f'train_fake_len = {len(fake_list[0: train_fake_idx])}')
        # print(f'test_real_len = {len(real_list[train_real_idx:])}')
        # print(f'test_fake_len = {len(fake_list[train_fake_idx:])}')
        print('Data loaded!')
    
    def load_kaggle_datasets(self):
        # load kaggle datasets
        with open("./datasets/waitan24/phase1/trainset_label.txt", "r") as f:
            lines = f.readlines()
            kaggle_train_real_data = []
            kaggle_train_fake_data = []
            kaggle_train_real_label = []
            kaggle_train_fake_label = []
            for line in lines[1:]:
                line = line.strip('\n').split(',')
                flag = int(line[-1])
                if flag == 1:
                    kaggle_train_fake_data.append('./datasets/waitan24/phase1/trainset/' + line[0])
                    kaggle_train_fake_label.append(1)
                else:
                    kaggle_train_real_data.append('./datasets/waitan24/phase1/trainset/' + line[0])
                    kaggle_train_real_label.append(0)
        kaggle_train_data = kaggle_train_real_data[0: 6000] + kaggle_train_fake_data[0: 6000]
        kaggle_train_label = kaggle_train_real_label[0: 6000] + kaggle_train_fake_label[0: 6000]
        kaggle_train_label_inv = [1 if item == 0 else 0 for item in kaggle_train_label]
        with open("./datasets/waitan24/phase1/valset_label.txt", "r") as f:
            lines = f.readlines()
            kaggle_val_data = []
            kaggle_val_label = []
            for line in lines[1:]:
                line = line.strip('\n').split(',')
                kaggle_val_data.append('./datasets/waitan24/phase1/valset/' + line[0])
                kaggle_val_label.append(int(line[1]))
        kaggle_val_data = kaggle_val_data[0: 4000]
        kaggle_val_label = kaggle_val_label[0: 4000]
        kaggle_val_label_inv = [1 if item == 0 else 0 for item in kaggle_val_label]
        return kaggle_train_data, kaggle_train_label_inv, kaggle_val_data, kaggle_val_label_inv
    
    def load_dfdc_datasets(self):
        # real_list = glob('./datasets/real-dfdc/*')
        # fake_list = glob('./datasets/fake-dfdc/*')
        # random.shuffle(real_list)
        # random.shuffle(fake_list)
        # real_tgt_list = [1] * len(real_list) # !!! important real label = 1 in our race
        # fake_tgt_list = [0] * len(fake_list) # !!! important fake label = 0 in our race
        # dfdc_train_data = real_list[0: 8000] + fake_list[0: 8000]
        # dfdc_train_label = real_tgt_list[0: 8000] + fake_tgt_list[0: 8000]
        # dfdc_val_data = real_list[8000: ] + fake_list[8000: ]
        # dfdc_val_label = real_tgt_list[8000: ] + fake_tgt_list[8000: ]
        # return dfdc_train_data, dfdc_train_label, dfdc_val_data, dfdc_val_label
        
        fake_list = glob('./datasets/fake-dfdc/*')
        random.shuffle(fake_list)
        fake_tgt_list = [0] * len(fake_list) # !!! important fake label = 0 in our race
        dfdc_train_data = fake_list[0: 8000]
        dfdc_train_label = fake_tgt_list[0: 8000]
        dfdc_val_data = fake_list[8000: ]
        dfdc_val_label = fake_tgt_list[8000: ]
        return dfdc_train_data, dfdc_train_label, dfdc_val_data, dfdc_val_label
    
    def load_celeb_datasets(self):
        real_list = glob('./datasets/real-celeb/*')
        fake_list = glob('./datasets/fake-celeb/*')
        random.shuffle(real_list)
        random.shuffle(fake_list)
        real_tgt_list = [1] * len(real_list) # !!! important real label = 1 in our race
        fake_tgt_list = [0] * len(fake_list) # !!! important fake label = 0 in our race
        celeb_train_data = real_list[0: 4000] + fake_list[0: 4000]
        celeb_train_label = real_tgt_list[0: 4000] + fake_tgt_list[0: 4000]
        celeb_val_data = real_list[4000: ] + fake_list[4000: ]
        celeb_val_label = real_tgt_list[4000: ] + fake_tgt_list[4000: ]
        return celeb_train_data, celeb_train_label, celeb_val_data, celeb_val_label
    
    def load_ff_datasets(self):
        real_list = glob('./datasets/real-ff/*')
        fake_list = glob('./datasets/fake-ff/*')
        random.shuffle(real_list)
        random.shuffle(fake_list)
        real_tgt_list = [1] * len(real_list) # !!! important real label = 1 in our race
        fake_tgt_list = [0] * len(fake_list) # !!! important fake label = 0 in our race
        ff_train_data = real_list[0: 4000] + fake_list[0: 4000]
        ff_train_label = real_tgt_list[0: 4000] + fake_tgt_list[0: 4000]
        ff_val_data = real_list[4000: ] + fake_list[4000: ]
        ff_val_label = real_tgt_list[4000: ] + fake_tgt_list[4000: ]
        return ff_train_data, ff_train_label, ff_val_data, ff_val_label
    
    def init_waitan(self):
        with open("./datasets/waitan24/phase1/trainset_label.txt", "r") as f:
            lines = f.readlines()
            train_data = []
            train_label = []
            for line in lines[1:]:
                line = line.strip('\n').split(',')
                train_data.append('./datasets/waitan24/phase1/trainset/' + line[0])
                train_label.append(int(line[1]))
        with open("./datasets/waitan24/phase1/valset_label.txt", "r") as f:
            lines = f.readlines()
            val_data = []
            val_label = []
            for line in lines[1:]:
                line = line.strip('\n').split(',')
                val_data.append('./datasets/waitan24/phase1/valset/' + line[0])
                val_label.append(int(line[1]))
        self.data['train'] = train_data
        self.labels['train'] = train_label
        self.data['val'] = val_data
        self.labels['val'] = val_label
        self.data['test'] = self.data['val']
        self.labels['test'] = self.labels['val']

    def init_cifar_10(self):
        
        import pickle
        _cifar_data = []
        _cifar_label = []
        for i in range(5):
            with open(f"../_Datasets/cifar-10-batches-py/data_batch_{i+1}", 'rb') as file:
                dict = pickle.load(file, encoding='bytes')
                tmp_data = np.array(dict[b'data'])
                tmp_label = np.array(dict[b'labels'])
                _cifar_data.append(tmp_data)
                _cifar_label.append(tmp_label)
        cifar_data = np.vstack([_cifar_data[i] for i in range(5)]).reshape(-1, 32, 32, 3)
        cifar_label = np.vstack([_cifar_label[i] for i in range(5)]).reshape(-1)

        self.data['train'] = cifar_data[0:40000]
        self.data['val'] = cifar_data[40000:45000]
        self.data['test'] = cifar_data[45000:50000]

        self.labels['train'] = cifar_label[0:40000]
        self.labels['val'] = cifar_label[40000:45000]
        self.labels['test'] = cifar_label[45000:50000]

    def init_celeb_df(self): 
        images_ids = self.__get_images_ids()
        test_ids = self.__get_test_ids()
        train_ids = [images_ids[0] - test_ids[0],
                     images_ids[1] - test_ids[1],
                     images_ids[2] - test_ids[2]]
        self.train_images, self.train_targets = self.__get_images(train_ids, balance=True)
        self.test_images, self.test_targets = self.__get_images(test_ids, balance=False)
        assert len(self.train_images) == len(self.train_targets) and \
            len(self.test_images) == len(self.test_targets), "The number of images and targets not consistent."
        
        self.data['train'] = self.train_images
        self.data['val'] = self.test_images
        self.data['test'] = self.test_images

        self.labels['train'] = self.train_targets
        self.labels['val'] = self.test_targets
        self.labels['test'] = self.test_targets
        print("Data from 'Celeb-DF' loaded.")

    def __get_images_ids(self):
        youtube_real = listdir(join(self.root, 'YouTube-real', 'images'))
        celeb_real = listdir(join(self.root, 'Celeb-real', 'images'))
        celeb_fake = listdir(join(self.root, 'Celeb-synthesis', 'images'))
        return set(youtube_real), set(celeb_real), set(celeb_fake)
    
    def __get_test_ids(self):
        youtube_real = set()
        celeb_real = set()
        celeb_fake = set()
        with open(join(self.root, "List_of_testing_videos.txt"), "r", encoding="utf-8") as f:
            contents = f.readlines()
            for line in contents:
                name = line.split(" ")[-1]
                number = name.split("/")[-1].split(".")[0]
                if "YouTube-real" in name:
                    youtube_real.add(number)
                elif "Celeb-real" in name:
                    celeb_real.add(number)
                elif "Celeb-synthesis" in name:
                    celeb_fake.add(number)
                else:
                    raise ValueError("'List_of_testing_videos.txt' file corrupted.")
        return youtube_real, celeb_real, celeb_fake
    
    def __get_images(self, ids, balance=False):
        real = list()
        fake = list()
        # YouTube-real
        for _ in ids[0]:
            real.extend(glob(join(self.root, 'YouTube-real', 'images', _, '*.png')))
        # Celeb-real
        for _ in ids[1]:
            real.extend(glob(join(self.root, 'Celeb-real', 'images', _, '*.png')))
        # Celeb-synthesis
        for _ in ids[2]:
            fake.extend(glob(join(self.root, 'Celeb-synthesis', 'images', _, '*.png')))
        print(f"Real: {len(real)}, Fake: {len(fake)}")
        if balance:
            fake = np.random.choice(fake, size=len(real), replace=False) # 在fake中随机抽取len个不重复元素
            print(f"After Balance | Real: {len(real)}, Fake: {len(fake)}")
        real_tgt = [0] * len(real)
        fake_tgt = [1] * len(fake)
        return [*real, *fake], [*real_tgt, *fake_tgt]

    def read_txt(self, oversample=False):

        dataset_files=[os.path.join(self.path,'test_fake.txt'), os.path.join(self.path,'test_real.txt'),
                        os.path.join(self.path,'val_fake.txt'), os.path.join(self.path,'val_real.txt'),
                        os.path.join(self.path,'train_fake.txt'), os.path.join(self.path,'train_real.txt')]
        # dataset_files= glob.glob(f"{self.path}")
        if 'FF++' in self.path or 'DFDC-Preview' in self.path:
            balance_ratio=4
        elif 'Celeb-DF-v2' in self.path:
            balance_ratio = 6
        elif 'DFDC' in self.path:
            balance_ratio = 5
        else:
            # unbalance fake and real
            balance_ratio = 1

        for file in dataset_files:
            file = file.replace('\\', '/') # path's diffenerce between linux and win
            with open(file, "r") as f:
                lines = f.readlines()
            if '/test_' in file:
                key='test'
            elif '/val_' in file:
                key='val'
            elif '/train_' in file:
                key = 'train'
            for i in range(len(lines)):
                # start clearing duplicates
                raw = re.sub("\s", "", lines[i]).split(",")
                paths = os.listdir(raw[1]) # video name, get file lists from Directory raw[1]
                for row in paths:
                    path_dir = os.path.join(raw[1], row).replace('\\', '/')
                    image_paths = os.listdir(path_dir)
                    for img_path in image_paths:
                        if '.png' in img_path:
                            if oversample and 'train_real' in file:
                                for i in range(balance_ratio):
                                    self.data[key].append(img_path)
                                    self.labels[key].append(int(raw[0]))
                            else:
                                self.data[key].append(img_path)
                                self.labels[key].append(int(raw[0])) 
                    # if os.path.isfile(path_dir) and '.png' in path_dir:
                    #     if oversample and 'train_real' in file:
                    #         for i in range(balance_ratio):
                    #             self.data[key].append(path_dir)
                    #             self.labels[key].append(int(raw[0]))
                    #     else:
                    #         self.data[key].append(path_dir)
                    #         self.labels[key].append(int(raw[0]))

    def get_dataset(self, mode):
        return self.data[mode], self.labels[mode]

class InferDataset():
    def __init__(self, args):
        self.data = {
            "train": [],
            "val":[],
            "test": [],
        }
        self.labels = copy.deepcopy(self.data)
        img_list = glob(args.data_path + '/*')
        self.data['train'] = img_list
        self.data['val'] = img_list
        self.data['test'] = img_list
        unkown_list = ['unk'] * len(img_list)
        self.labels['train'] = unkown_list
        self.labels['val'] = unkown_list
        self.labels['test'] = unkown_list
        print('Inference data loaded')

class MyDataset(Dataset):

    def __init__(self,
                 args,
                 data,
                 label,
                 size=224,
                 normalize={"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]},
                 test=False):
        super().__init__()
        self.args = args
        self.size = size
        self.data = data
        self.label = label
        self.normalize = normalize
        self.aug = self.create_train_aug()
        self.transform = self.transform_all()
        self.test = test
        # if self.args.extract_face:
        #     self.retainFace = Retain_Face()

    def create_train_aug(self):
        return Compose([
            ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            GaussNoise(p=0.1),
            GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(),
            PadIfNeeded(min_height=self.size, min_width=self.size, border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
        )
    
    def transform_all(self):
        return Resize(p=1, height=self.size, width=self.size)
    
    def __getitem__(self, idx):
        if 'cifar' not in self.args.dataset:
            # img = Image.open(self.data[idx])
            img = cv2.imread(self.data[idx], cv2.IMREAD_COLOR) # HxWx3
            # if self.args.extract_face:
            #     with torch.no_grad():
            #         img = self.retainFace.generateBooxImg(img)
            data = self.transform(image=img) # H*W*3
            img = data["image"]
            if not self.test:
                data = self.aug(image=img)
                img = data["image"]
            img = img_to_tensor(img, self.normalize) # 3*H*W
            return img, self.label[idx] # 3*H_*W_=224
        else:
            img = self.data[idx]
            # if not self.test:
            #     data = cifar_transform_train(image=img)
            #     img = data["image"]
            # img = cifar_transform_train(Image.fromarray(img))
            # img = img_to_tensor(img, self.normalize) # 3*H*W
            return img, self.label[idx] # 3*H_*W_=224

    def __len__(self):
        return len(self.data)

# mixup_data(data, y, self.args.alpha)
def mixup_data(x, y, alpha=0.5, use_cuda=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size) # 将[0, batch_size)中的元素随机排列

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    a = criterion(pred, y_a)
    b = criterion(pred, y_b)
    losses = []
    try:
        for i in range(len(a)):
            losses.append(lam * a[i]  + (1 - lam) * b[i])
    except:
        return lam * a  + (1 - lam) * b
    return losses
