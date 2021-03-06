import albumentations as A
from albumentations.pytorch import ToTensorV2, ToTensor


class BasicAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            ToTensorV2()
        ])

    def __call__(self, image):
        return self.transform(image=image)


class ImagenetDefaultAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __call__(self, **data):
        return self.transform(**data)


class CustomAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(0.5),
            A.VerticalFlip(0.5),
            A.Rotate(p=0.5),
            ToTensorV2(),
        ])

    def __call__(self, **data):
        return self.transform(**data)