import torch
import torch.utils.data as D
from torchvision import transforms
import random
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np



class RandomHorizontalFlip1(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, imageA,imageB,label):
        if random.random() < self.flip_prob:
            imageA = F.hflip(imageA)
            imageB = F.hflip(imageB)
            label = F.hflip(label)
        return imageA,imageB,label


class RandomVerticalFlip1(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, imageA,imageB,label):
        if random.random() < self.flip_prob:
            imageA = F.vflip(imageA)
            imageB = F.vflip(imageB)
            label = F.vflip(label)
        return imageA,imageB,label

class Normalize111(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self,imageA,imageB,label):
        imageA = F.normalize(imageA, self.mean, self.std, self.inplace)
        imageB = F.normalize(imageB, self.mean, self.std, self.inplace)
        return imageA,imageB,label

class ToTensor1(object):
    def __call__(self, imageA,imageB,label):
        imgA = F.to_tensor(imageA)
        imgB = F.to_tensor(imageB)
        label = F.to_tensor(label)
        return imgA,imgB,label


class Compose1(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imageA,imageB,label):
        for t in self.transforms:
            imageA,imageB,label = t(imageA,imageB,label)

        return imageA,imageB,label

class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def addZao(self,img):

        img_ = np.array(img).copy()
        h, w, c = img_.shape

        signal_pct = self.snr

        noise_pct = (1 - self.snr)
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])

        mask = np.repeat(mask, c, axis=2)
        img_[mask == 1] = 255  # 盐噪声
        img_[mask == 2] = 0  # 椒噪声
        return Image.fromarray(img_.astype('uint8')).convert('RGB')  # 转化为PIL的形式

    def __call__(self, imageA,imageB,label):
        if random.uniform(0, 1) < self.p: # 按概率进行
            imageA = self.addZao(imageA)
            imageB = self.addZao(imageB)
            return imageA,imageB,label
        else:
            return imageA,imageB,label



def load_img():
    data_transforms = {#进行数据增强，但是需要A、B、label一起变换
        'train': Compose1([
            RandomHorizontalFlip1(0.5),  # 水平翻转
            RandomVerticalFlip1(0.5),  # 垂直翻转
            # AddPepperNoise(0.5),
            ToTensor1(), 
            Normalize111((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'test': Compose1([
            ToTensor1(), 
            Normalize111((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'val': Compose1([
            ToTensor1(),  
            Normalize111((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    class MyDataset(Dataset):
        def __init__(self, img_path, transform):
            super(MyDataset, self).__init__()
            self.root = 'D:\Desktop\dataset\levircd11/list'+img_path
            f = open(self.root, 'r')
            data = f.read().splitlines()
            imgA = [] 
            imgB = []
            label = []
            for line in data: 
                imgA.append('D:\Desktop\dataset\levircd11/A'+line)
                imgB.append('D:\Desktop\dataset\levircd11/B'+line)
                label.append('D:\Desktop\dataset\levircd11/label'+line)
            self.imgA = imgA
            self.imgB = imgB
            self.label = label
            self.transform = transform
        def __len__(self):
            return len(self.label)

        def __getitem__(self, item):
            imgA =self.imgA[item]
            imgB = self.imgB[item]
            label = self.label[item]
            imgA = Image.open(imgA).convert('RGB')
            imgB = Image.open(imgB).convert('RGB')
            label = Image.open(label).convert('1')
            imgA,imgB,label = self.transform(imgA,imgB,label)
            return imgA,imgB,label


    image_datasets = {}

    for x in ['train','test','val']:
        name = x+'.txt'
        image_datasets[x] = MyDataset(name, data_transforms[x])

    batch_size = 1 
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'test', 'val']}

    traindataloader = dataloaders['train']
    testdataloader = dataloaders['test']
    valdataloader = dataloaders['val']

    return traindataloader,testdataloader,valdataloader,batch_size

if __name__ == '__main__':
    traindataloader,testdataloader,valdataloader,batch_size = load_img()

    for batch_idx,(input1,input2,label) in enumerate(valdataloader):
        print('input1的shape',input1.size())
        print(type(input1))
        x = input1[0]
        y = input2[0]
        toPIL = transforms.ToPILImage()
        pic = toPIL(x)
        pic.save('random.jpg')
        pic = toPIL(y)
        pic.save('random1.jpg')
        pic = toPIL(label[0])
        pic.save('random2.jpg')
        break
