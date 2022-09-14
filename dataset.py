import albumentations
import torch

import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationDataset:
    def __init__(self, image_paths, targets, resize=None):
        # resize = (height, width)
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.aug = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                )
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        targets_1, targets_2, targets_3 = [t[item] for t in self.targets]
        # print('----------')
        # print(image.size)
        # print(f'=== {targets_1} {targets_2} {targets_3}')
        init_len = len(targets_1)
        pad_len = 20 - init_len
        targets_1 = torch.concat([torch.tensor(targets_1, dtype=torch.long), torch.ones([pad_len])])
        targets_2 = torch.concat([torch.tensor(targets_2, dtype=torch.long), torch.ones([pad_len])])
        targets_3 = torch.concat([torch.tensor(targets_3, dtype=torch.long), torch.ones([pad_len])])

        if self.resize is not None:
            w, h = image.size
            image = image.resize(
                (int(self.resize[0] * w / h), self.resize[0]), resample=Image.BILINEAR
            )
            new_image = Image.new(image.mode, (self.resize[1], self.resize[0]), (0, 0, 0))
            new_image.paste(image)
            image = new_image

        image = np.array(image)
        augmented = self.aug(image=image)
        image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets_1": targets_1,
            "targets_2": targets_2,
            "targets_3": targets_3,
            "lengths": torch.tensor(init_len, dtype=torch.int32)
        }
