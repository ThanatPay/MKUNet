import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, VerticalFlip, Rotate

def load_data():
    # Load the images and masks
    simple_ptc_images = sorted(glob("/content/drive/MyDrive/Thyroid/FinalData/dataset/PTC_image/*.png"))
    simple_ptc_masks = sorted(glob("/content/drive/MyDrive/Thyroid/FinalData/dataset/PTC_mask/*.png"))
    difficult_ptc_images = sorted(glob("/content/drive/MyDrive/Thyroid/FinalData/dataset/Small_PTC_image/*.png"))
    difficult_ptc_masks = sorted(glob("/content/drive/MyDrive/Thyroid/FinalData/dataset/Small_PTC_mask/*.png"))
    benign_images = sorted(glob("/content/drive/MyDrive/Thyroid/FinalData/dataset/Benign_image/*.jpg"))
    benign_masks = sorted(glob("/content/drive/MyDrive/Thyroid/FinalData/dataset/Benign_mask.jpg"))*len(benign_images)

    # Split the data
    train_simple_ptc_images, test_simple_ptc_images = train_test_split(simple_ptc_images, test_size=106, random_state=42)
    val_simple_ptc_images, test_simple_ptc_images = train_test_split(test_simple_ptc_images, test_size=0.5, random_state=42)

    train_simple_ptc_masks, test_simple_ptc_masks = train_test_split(simple_ptc_masks, test_size=106, random_state=42)
    val_simple_ptc_masks, test_simple_ptc_masks = train_test_split(test_simple_ptc_masks, test_size=0.5, random_state=42)

    train_difficult_ptc_images, test_difficult_ptc_images = train_test_split(difficult_ptc_images, test_size=14, random_state=42)
    val_difficult_ptc_images, test_difficult_ptc_images = train_test_split(test_difficult_ptc_images, test_size=0.5, random_state=42)

    train_difficult_ptc_masks, test_difficult_ptc_masks = train_test_split(difficult_ptc_masks, test_size=14, random_state=42)
    val_difficult_ptc_masks, test_difficult_ptc_masks = train_test_split(test_difficult_ptc_masks, test_size=0.5, random_state=42)

    train_benign_images, test_benign_images = train_test_split(benign_images, test_size=120, random_state=42)
    val_benign_images, test_benign_images = train_test_split(test_benign_images, test_size=0.5, random_state=42)

    train_benign_masks, test_benign_masks = train_test_split(benign_masks, test_size=120, random_state=42)
    val_benign_masks, test_benign_masks = train_test_split(test_benign_masks, test_size=0.5, random_state=42)

    train_x = train_simple_ptc_images + train_difficult_ptc_images + train_benign_images
    val_x = val_simple_ptc_images + val_difficult_ptc_images + val_benign_images
    test_x = test_simple_ptc_images + test_difficult_ptc_images + test_benign_images
    train_y = train_simple_ptc_masks + train_difficult_ptc_masks + train_benign_masks
    val_y = val_simple_ptc_masks + val_difficult_ptc_masks + val_benign_masks
    test_y = test_simple_ptc_masks + test_difficult_ptc_masks + test_benign_masks

    return train_x, train_y, val_x, val_y, test_x, test_y

def augment_data(images, masks, save_path, augment=True):
    """ Performing data augmentation. """
    H = 512
    W = 512

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the dir name and image name """
        dir_name = x.split("/")[-2]
        name = dir_name + "_" + x.split("/")[-1].split(".")[0]
        # Read the image and mask

        # '''note
        # x = cv2.imread(x, cv2.IMREAD_COLOR)
        # y = cv2.imread(y, cv2.IMREAD_COLOR)
        # '''
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        (t,blackwhite) = cv2.threshold(y, 5, 255, cv2.THRESH_BINARY)
        y = blackwhite

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]

        else:
            X = [x]
            Y = [y]

        for i, (img, mk) in enumerate(zip(X, Y)):
            img = cv2.resize(img, (W, H))
            mk = cv2.resize(mk, (W, H))
            mk = mk/255.0
            mk = (mk > 0.5) * 255

            if len(X) == 1:
                tmp_image_name = f"{name}.jpg"
                tmp_mask_name  = f"{name}.jpg"
            else:
                tmp_image_name = f"{name}_{i}.jpg"
                tmp_mask_name  = f"{name}_{i}.jpg"

            image_path = os.path.join(save_path, "image/", tmp_image_name)
            mask_path  = os.path.join(save_path, "mask/", tmp_mask_name)

            cv2.imwrite(image_path, img)
            cv2.imwrite(mask_path, mk)

# train_x, train_y, val_x, val_y, test_x, test_y = load_data()
# augment_data(train_x, train_y, "new_data/train/", augment=True)
# augment_data(val_x, val_y, "new_data/val/", augment=True)
# augment_data(test_x, test_y, "new_data/test/", augment=False)