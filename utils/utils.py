from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import os


class ImageFolderDataset(Dataset):
    def __init__(self,root,transform=None):
        super(ImageFolderDataset,self).__init__()
        self.root=root
        self.transform=transform
        self.files=list(os.listdir(root))
        self.files=[p for p in self.files if p.endswith(('.jpeg','.jpg','png'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        image_path = os.path.join(
            self.root,
            self.files[idx]
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

def get_transform(size,crop,final_size):
    transfrom_list=[]

    if size>0:
        transfrom_list.append(transforms.Resize(size))

    if crop:
        transfrom_list.append(transforms.RandomCrop(final_size))

    else:
        transfrom_list.append(transforms.Resize(final_size))
    transfrom_list.append(transforms.ToTensor())

    return transforms.Compose(transfrom_list)


def cal_mean_std(feature,eps=1e-5):
    size=feature.size()   #[bacth_size,channnels,h,w]
    assert (len(size)==4)
    batch_size,channels=size[:2]
    feature_mean=feature.view(batch_size,channels,-1).mean(dim=2).view(batch_size,channels,1,1)
    feature_var=feature.view(batch_size,channels,-1).var(dim=2,unbiased=False)+eps
    feature_std=feature_var.sqrt().view(batch_size,channels,1,1)
    return feature_mean,feature_std

def adaptive_instance_normalization(content_features,styel_features):
    size=content_features.size()
    style_mean,style_std=cal_mean_std(styel_features)
    content_mean,content_std=cal_mean_std(content_features)
    normalized_content_features=(content_features-content_mean.expand(size))/content_std.expand(size)
    return normalized_content_features*style_std.expand(size)+style_mean.expand(size)