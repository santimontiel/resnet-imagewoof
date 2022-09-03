import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from random import randint
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from matplotlib import pyplot as plt

class ImageWoof(Dataset):

    FOLDER_TO_CLS = {
        "n02093754": 0, 
        "n02089973": 1, 
        "n02099601": 2, 
        "n02087394": 3, 
        "n02105641": 4, 
        "n02096294": 5, 
        "n02088364": 6, 
        "n02115641": 7, 
        "n02111889": 8, 
        "n02086240": 9,
    }

    CLS_TO_LABEL = {
        0: "Australian terrier",
        1: "Border terrier",
        2: "Samoyed",
        3: "Beagle",
        4: "Shih-Tzu",
        5: "English foxhound",
        6: "Rhodesian ridgeback",
        7: "Dingo",
        8: "Golden retriever",
        9: "Old English sheepdog"
    }

    def __init__(self,
                 path_to_root: Path,
                 is_train: bool,
                 transforms: Optional[transforms.Compose] = None,
                ) -> None:
        """
        Args:
            - path_to_root [Path]
                A path to the folder where the dataset is stored.
            - is_train [bool]
                A boolean to chose the train split if True, val split if False.
            - transforms [Optional[transforms.Compose]]
                Transforms applied to the image.
        """        
        self.train_or_val = "train" if is_train else "val"
        self.img_paths = list((path_to_root / self.train_or_val).glob("*/*.JPEG"))
        self.transforms = transforms

    def _get_x(self, index: int) -> Image:
        img_path = self.img_paths[index]
        image = Image.open(img_path).convert('RGB')
        return image

    def _get_y(self, index: int) -> int:
        img_path = self.img_paths[index]
        cls = self.FOLDER_TO_CLS.get(img_path.parent.stem)
        return cls

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self._get_x(index)
        cls = self._get_y(index)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, cls

    def __len__(self) -> int:
        return len(self.img_paths)

    def show_batch(self) -> None:
        # @TODO: Send this function to visualize utils. Should then receive
        #           dataset as a parameter. (Dataset-agnostic?)

        # Get a sorted list of 8 unique random indexes contained on the dataset.
        random_idxs = []
        for i in range(0, 8):
            tmp = randint(0, self.__len__()) 
            while tmp in random_idxs:
                tmp = randint(0, self.__len__())
            random_idxs.append(tmp)
        random_idxs.sort()
            
        # Get 8 tuples of image, label.
        batch = [self.__getitem__(random_idxs[i]) for i in range(0, 8)]
        
        # Generate a 4x2 mosaic grid.
        import matplotlib.pyplot as plt
        fig, axd = plt.subplot_mosaic([
            ["00", "01", "02", "03"],
            ["10", "11", "12", "13"]
            ], figsize=(8, 4), constrained_layout=True)

        for i, k in enumerate(axd):
            axd[k].imshow(batch[i][0].permute(1,2,0))
            axd[k].set_title(self.CLS_TO_LABEL.get(batch[i][1]), fontsize=8)
        fig.suptitle("ImageWoof Dataset batch")


def image_woof_train_dataloader(path_to_root, bs, transforms) -> DataLoader:
    return DataLoader(
        dataset=ImageWoof(path_to_root=path_to_root, is_train=True, transforms=transforms),
        batch_size=bs,
        shuffle=True
    )

def image_woof_test_dataloader(path_to_root, bs, size) -> DataLoader:
    tfs = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor()
    ])  
    return DataLoader(
        dataset=ImageWoof(path_to_root=path_to_root, is_train=False, transforms=tfs),
        batch_size=bs,
        shuffle=False,
    )