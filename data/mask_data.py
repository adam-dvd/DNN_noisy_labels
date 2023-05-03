"""Script extracted from
Code for ICCV2019 Paper ["O2U-Net: A Simple Noisy Label Detection Approach for Deep Neural Networks"](https://ieeexplore.ieee.org/document/9008796)
@INPROCEEDINGS{huang2019o2unet,
  author={Huang, Jinchi and Qu, Lie and Jia, Rongfei and Zhao, Binqiang},
  booktitle={2019 IEEE/CVF International Conference on Computer Vision (ICCV)}, 
  title={O2U-Net: A Simple Noisy Label Detection Approach for Deep Neural Networks}, 
  year={2019},
  pages={3325-3333},
  doi={10.1109/ICCV.2019.00342},
  }
"""


from  torch.utils import data
from  PIL import  Image
from  io import BytesIO
class Mask_Select(data.Dataset):
    def __init__(self, origin_dataset,mask_index):
        self.transform = origin_dataset.transform
        self.target_transform = origin_dataset.target_transform
        labels=origin_dataset.train_noisy_labels
        dataset=origin_dataset.train_data
        self.dataname=origin_dataset.dataset
        self.origin_dataset=origin_dataset
        self.train_data=[]
        self.train_noisy_labels=[]
        for i,m in enumerate(mask_index):
            if m<0.5:
                continue
            self.train_data.append(dataset[i])
            self.train_noisy_labels.append(labels[i])


        print ("origin set number:%d"%len(labels),"after clean number:%d" % len(self.train_noisy_labels))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.train_data[index], self.train_noisy_labels[index]

        if self.dataname!='MinImagenet':
            img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.train_data)
