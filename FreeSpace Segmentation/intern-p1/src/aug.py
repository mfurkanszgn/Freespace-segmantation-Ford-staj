from numpy import asarray
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from constant import *
import tqdm
import albumentations as A
import Automold as am
import Helpers as hp
import cv2 
from torchvision import transforms as T
from PIL import Image
A_MASK_DIR="../data/augmentation_mask"
A_IMAGE_DIR="../data/augmentations"
A_IMAGE_OUT_DIR='../data/masked_augimages'





#The path to the masks folder is assigned to the variable
image_path=[] #empty list created
for name in os.listdir(IMAGE_DIR):
    image_path.append(os.path.join(IMAGE_DIR,name))
mask_path=[] #empty list created


for name in os.listdir(MASK_DIR):
    mask_path.append(os.path.join(MASK_DIR,name))
    
    
valid_size = 0.3
test_size  = 0.1
indices = np.random.permutation(len(image_path))
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)
train_input_path_list = image_path#We got the elements of the image_path_list list from 1905 to the last element
train_label_path_list = mask_path#We got the elements of the mask_path_list list from 1905 to the last element
print(len(train_input_path_list))
print(len(train_label_path_list))

transform = A.Compose([
    #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.8, p=1),
   
   
    #A.HorizontalFlip(True,p=1)
    A.RandomShadow (shadow_roi=(0, 0.1, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=1)
])
"""
just_Flip=A.Compose([
     A.HorizontalFlip(True,p=1)
    
    ])
#seq kullanım
"""
"""
def augment_shadow(img):
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    shadow_mask = 0 * hsv[:, :, 1]
    X_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][0]
    Y_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][1]

    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1

    shadow_density = .5
    left_side = shadow_mask == 1
    right_side = shadow_mask == 0

    if np.random.randint(2) == 1:
        hsv[:, :, 2][left_side] = hsv[:, :, 2][left_side] * shadow_density
    else:
        hsv[:, :, 2][right_side] = hsv[:, :, 2][right_side] * shadow_density

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


"""
"""
# random shadow1
for image in tqdm.tqdm(train_input_path_list):
    images_aug=cv2.imread(image)
    images_aug=np.array(images_aug).astype(np.uint8)
    
    new_image_path=image[:-4]+"-1"+".jpg"
    new_image_path=new_image_path.replace('image', 'augmentation')
    
    images_aug=augment_shadow(images_aug)
    
    cv2.imwrite(new_image_path,images_aug)

        
window_name = 'image'
cv2.imshow(window_name, images_aug)
cv2.waitKey(0) 
plt.imshow(images_aug)
cv2.destroyAllWindows()     


"""




for image in tqdm.tqdm(train_input_path_list):
    images_aug=cv2.imread(image)
    images_aug=np.array(images_aug).astype(np.uint8)
    
    new_image_path=image[:-4]+"-1"+".jpg"
    new_image_path=new_image_path.replace('image', 'augmentation')
    images_aug=transformed_image_1 = transform(image=images_aug)['image']
   
    
    
   
    cv2.imwrite(new_image_path,images_aug)

        
window_name = 'image'
cv2.imshow(window_name, images_aug)
cv2.waitKey(0) 
plt.imshow(images_aug)
cv2.destroyAllWindows()     











"""
#color jitter 
for image in tqdm.tqdm(train_input_path_list):
    img=Image.open(image)
    color_aug = T.ColorJitter(brightness=0.4, contrast=0.4, hue=0.06)

    img_aug = color_aug(img)
    new_path=image[:-4]+"-1"+".jpg"
    new_path=new_path.replace('image', 'augmentation')
    img_aug=np.array(img_aug)
    cv2.imwrite(new_path,img_aug)
"""





for mask in tqdm.tqdm(train_label_path_list):
    msk=cv2.imread(mask,0)
    msk=np.array(msk).astype(np.uint8)
    newm_path=mask[:-4]+"-1"+".png"
    newm_path=newm_path.replace('masks', 'augmentation_mask')
    #msk=transformed_image_1 = just_Flip(image=msk)['image']
    cv2.imwrite(newm_path,msk)


# For mask on image 

# Create a list which contains every file name in masks folder
mask_list = os.listdir(A_MASK_DIR)
# Remove hidden files if any
for f in mask_list:
    if f.startswith('.'):
        mask_list.remove(f)

# For every mask image
for mask_name in tqdm.tqdm(mask_list):
    # Name without extension
    mask_name_without_ex = mask_name.split('.')[0]
   # print(mask_name_without_ex)
    # Access required folders
    mask_path      = os.path.join(A_MASK_DIR, mask_name)
    image_path     = os.path.join(A_IMAGE_DIR, mask_name_without_ex+'.jpg')
    image_out_path = os.path.join(A_IMAGE_OUT_DIR, mask_name)

    # Read mask and corresponding original image
    mask  = cv2.imread(mask_path, 0).astype(np.uint8)
    image = cv2.imread(image_path).astype(np.uint8)
    
    # Change the color of the pixels on the original image that corresponds
    # to the mask part and create new image
    cpy_image  = image.copy()
    image[mask==1, :] = (255, 0, 125)
    
    opac_image = (image/2 + cpy_image/2).astype(np.uint8)
    
    # Write output image into IMAGE_OUT_DIR folder
    cv2.imwrite(image_out_path, opac_image)

    # Visualize created image if VISUALIZE option is chosen
    """if VISUALIZE:
        plt.figure()
        plt.imshow(opac_image)
        plt.show()"""


