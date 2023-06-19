import torch as th
from torchvision.datasets import ImageNet
from torchvision import transforms
import math
from PIL import Image
import numpy as np

# path = "/data/imagenet"
# # batch_size = 50
# img_size = 256


# imagenet_labels = [classes[0] for classes in dataset.classes]

cached_dataset = None

def unload_imagenet_dataset():
    global cached_dataset
    cached_dataset = None

def get_dataset(imagenet_path, img_size):
    # Model/ds specific transforms
    transform_list = [
        transforms.Resize(int(math.floor(img_size / 0.875)), interpolation=Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ]
    transform = transforms.Compose(transform_list)
    global cached_dataset
    if cached_dataset is None:
        print("Load imagenet")
        dataset = ImageNet(imagenet_path, split='val', transform=transform)
        cached_dataset = dataset
    else:
        dataset = cached_dataset
    return dataset


def get_imagenet_class_name(imagenet_path, class_ids):
    return [get_dataset(imagenet_path, 256).classes[class_id][0] for class_id in class_ids]

def load_imagenet_class(imagenet_path, img_size, input_class, target_class):
    dataset = get_dataset(imagenet_path, img_size)

    image_idx = [id for id, (path, cls) in enumerate(dataset.samples) if cls == input_class]
    # You can use this to select a subset of images
    # image_idx = image_idx[:4]

    some_vces = {idx: [target_class] for idx in image_idx}
    return load_selected_imagenet_samples(imagenet_path, img_size, some_vces, dataset)


def load_selected_imagenet_samples(imagenet_path, img_size, some_vces=None, dataset=None):
    if dataset is None:
        dataset = get_dataset(imagenet_path, img_size)

    # Pre-selected images and VCE targets from https://github.com/valentyn1boreiko/DVCEs
    if some_vces is None:
        some_vces = {
            14655: [288, 292],
            10452: [207, 208],
            46751: [924, 959],
            48679: [970, 972],
            48539: [970, 980],
            48282: [963, 965]
        }

    imgs, labels, targets = [], [], []
    for img_idx, img_targets in some_vces.items():
        for target in img_targets:
            img, label = dataset[img_idx]
            imgs.append(img)
            labels.append(label)
            targets.append(target)

    assert len(imgs) == len(labels) == len(targets)
    imgs = th.stack(imgs).mul(2).sub(1)  # .clone()
    labels = th.tensor(labels)
    targets = th.tensor(targets)
    
    target_imgs = th.stack([dataset[img_idx][0] for img_idx in np.where(np.array(dataset.targets)==targets[0].item())[0]]).mul(2).sub(1)
    
    return {"images": imgs, "labels": labels, "targets": targets, "target_imgs": target_imgs}

# in_loader = dl.get_ImageNet(path=hps.data_folder, train=False, augm_type='crop_0.875', size=img_size)
# in_dataset = in_loader.dataset
# loader = th.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                          shuffle=False, num_workers=8)
# accepted_wnids = []
# num_imgs = 50

# imgs = th.zeros((num_imgs, 3, img_size, img_size))
# segmentations = th.zeros((num_imgs, 3, img_size, img_size))
# targets_tensor = th.zeros(num_imgs, dtype=th.long)
# labels_tensor = th.zeros(num_imgs, dtype=th.long)
# filenames = []

# image_idx = 0


# for i, (img_idx, target_classes) in enumerate(list(some_vces.items())):
#     print(i,img_idx)
#     in_image, label = dataset[img_idx]
#     for i in range(len(target_classes)):
#         targets_tensor[image_idx+i] = target_classes[i]
#         labels_tensor[image_idx+i] = label
#         imgs[image_idx+i] = in_image
#     image_idx += len(target_classes)
#     if image_idx >= num_imgs:
#         break
# print()
