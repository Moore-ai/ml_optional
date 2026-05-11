from torchvision import transforms
from torchvision.transforms import InterpolationMode

from classify_Hyper_Kvasir.config import IMG_SIZE, USE_RAND_AUGMENT, RA_NUM_OPS, RA_MAGNITUDE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_train_augmentation = (
    transforms.RandAugment(num_ops=RA_NUM_OPS, magnitude=RA_MAGNITUDE)
    if USE_RAND_AUGMENT
    else transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])
)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
    _train_augmentation,
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

test_transform = val_transform
