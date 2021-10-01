import json
import os
import numpy as np
import cv2
import tqdm
from constant import JSON_DIR, MASK_DIR



# Create a list which contains every file name in "jsons" folder
json_list = os.listdir(JSON_DIR)  # json dosylarini bir listeye doldurduk.
print(json_list)
""" tqdm Example Start"""

iterator_example = range(1000000)

for i in tqdm.tqdm(iterator_example): # progress bar icin kullanildi.
    pass

""" rqdm Example End"""


# For every json file
for json_name in tqdm.tqdm(json_list):   #her bir json dosyası okunur ve icerisindeki freesapce noktalari belirlenerek maska yani png dosylarının uzerine islenir.
                            
    # Access and open json file as dictionary
    json_path = os.path.join(JSON_DIR, json_name) # json dosyasinin tam adresi icin
    json_file = open(json_path, 'r')

    # Load json data
    json_dict = json.load(json_file)  # json dosyasini bir dict haline getiriyor ki okuyabilelim.

    # Create an empty mask whose size is the same as the original image's size
    mask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8) # normal resimlerin boyutlari kadar bir mask png olusturur.
    
    mask_path = os.path.join(MASK_DIR, json_name[:-9]+".png")

    # For every objects
    for obj in json_dict["objects"]:
        # Check the objects ‘classTitle’ is ‘Freespace’ or not.
        if obj['classTitle']=='Freespace':
            # Extract exterior points which is a point list that contains
            # every edge of polygon and fill the mask with the array.
            mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]), color=1)

    # Write mask image into MASK_DIR folder
    cv2.imwrite(mask_path, mask.astype(np.uint8))  # mask png kaydedilir.
