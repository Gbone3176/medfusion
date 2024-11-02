import torch.utils.data as data 
import torch 
from torch import nn
from pathlib import Path 
from torchvision import transforms as T
import pandas as pd 
import numpy as np

from PIL import Image

from medical_diffusion.data.augmentation.augmentations_2d import Normalize, ToTensor16bit

class SimpleDataset2D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers =[],
        crawler_ext = 'jpg', # other options are ['tif', 'jpeg', 'png', 'tiff'],
        transform = None,
        image_resize = None,
        augment_horizontal_flip = False,
        augment_vertical_flip = False, 
        image_crop = None,
    ):
        super().__init__()
        self.path_root = Path(path_root)
        self.crawler_ext = crawler_ext
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_ext) 

        if transform is None: 
            self.transform = T.Compose([
                T.Resize(image_resize) if image_resize is not None else nn.Identity(),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                T.CenterCrop(image_crop) if image_crop is not None else nn.Identity(),
                T.ToTensor(),
                # T.Lambda(lambda x: torch.cat([x]*3) if x.shape[0]==1 else x),
                # ToTensor16bit(),
                # Normalize(), # [0, 1.0]
                # T.ConvertImageDtype(torch.float),
                T.Normalize(mean=0.5, std=0.5) # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        # img = Image.open(path_item) 
        img = self.load_item(path_item)
        return {'uid':rel_path_item.stem, 'source': self.transform(img)}
    
    def load_item(self, path_item):
        return Image.open(path_item).convert('RGB') 
        # return cv2.imread(str(path_item), cv2.IMREAD_UNCHANGED) # NOTE: Only CV2 supports 16bit RGB images 
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [path.relative_to(path_root) for path in Path(path_root).rglob(f'*.{extension}')]

    def get_weights(self):
        """Return list of class-weights for WeightedSampling"""
        return None 


class ISIC2018_Dataset(SimpleDataset2D):
    def __init__(self, path_root, transform=None):
        super().__init__(path_root, transform=transform)
        self.path_root = Path(path_root)
        self.labels = pd.read_csv(self.path_root / 'TrainingGroundTruth.csv', index_col='image')
        self.labels = self.labels.astype(float)
        self.target_list = []
        for index, row in self.labels.iterrows():
            self.target_list.append(np.argmax(row.values))
        
    def __len__(self):
        return len(self.target_list)
    

    def __getitem__(self, index):
        image_name = self.labels.index[index]
        path_item = self.path_root / 'train_data' / f'{image_name}.jpg'
        img = Image.open(path_item).convert('RGB')

        if self.transform:
            img = self.transform(img)

        target = self.target_list[index]

        # target = np.argmax(self.labels.loc[image_name].values, dtype=torch.float32)

        # target_list = torch.tensor(self.labels.loc[image_name].values, dtype=torch.float32)
        # first_index = torch.nonzero(target_list == 1.0, as_tuple=False)
        # if first_index.numel() > 0:
        #     target = first_index[0].item()
        # else:
        #     target = None

        return {'uid': image_name, 'source': img, 'target': target}

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

    def get_weights(self):
        n_samples = len(self.target_list)

        class_counts = np.bincount(self.target_list)

        weight_per_class = np.round(n_samples / class_counts, 2)  # 使用总样本数和类别数计算权重

        # 初始化权重列表
        weights = [0] * n_samples
        for index in range(n_samples):
            target = self.target_list[index]  # 获取当前样本的类别索引
            weights[index] = weight_per_class[target]  # 分配对应类别的权重

        return weights 

if __name__ == "__main__":
    # 确保您已设置好路径和数据集结构
    path_root = '/storage/ScientificPrograms/Conditional_Diffusion/ISIC_data/ISIC2018'  # 替换为实际路径
    transform = T.Compose([
        T.Resize((128, 128)),  # 示例图像大小
        T.ToTensor(),
    ])

    # 初始化数据集
    dataset = ISIC2018_Dataset(path_root, transform=transform)

    # 测试数据集长度
    print(f"Total samples: {len(dataset)}")

    # 测试获取一个样本
    sample = dataset[0]
    print(f"Sample UID: {sample['uid']}")
    print(f"Image tensor shape: {sample['source'].shape}")
    print(f"Target: {sample['target']}")

    # 测试权重计算
    weights = dataset.get_weights()
    print(f"weights:{weights[:100]}")

    # 验证权重数量与样本数量是否一致
    print(len(dataset))
    assert len(weights) == len(dataset), "Weight length should match dataset length."

    # 验证权重的类型
    for weight in weights:
        assert isinstance(weight, float), "Weights should be floats."

    print("All tests passed successfully!")