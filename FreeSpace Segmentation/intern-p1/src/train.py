from model import FoInternNet
from UnetModel import UNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, ConcatDataset
from random import shuffle
import cv2
from sklearn.model_selection import KFold
######### PARAMETERS ##########
valid_size = 0.3
test_size  = 0.1
batch_size =6
epochs =1
cuda = False
input_shape = (224, 224)
n_classes = 2


    
###############################

######### DIRECTORIES #########
SRC_DIR = os.getcwd()

ROOT_DIR = os.path.join(SRC_DIR, '..')
#print("root dir:", ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

IMAGE_DIR = os.path.join(DATA_DIR, 'images')

MASK_DIR = os.path.join(DATA_DIR, 'masks')

AUG_IMAGE=os.path.join(DATA_DIR,'augmentations')
AUG_MASK=os.path.join(DATA_DIR,'augmentation_mask')
###############################

# PREPARE IMAGE AND MASK LISTS

    
aug_path_list = glob.glob(os.path.join(AUG_IMAGE, '*'))
aug_path_list.sort()
aug_mask_path_list = glob.glob(os.path.join(AUG_MASK, '*'))
aug_mask_path_list.sort()


image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))

image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()


image_path_list= image_path_list#+aug_path_list
mask_path_list= mask_path_list#+aug_mask_path_list



# Data shuffle
N = len(image_path_list)
ind_list = [i for i in range(N)]
shuffle(ind_list)

image_path_list  = list(np.array(image_path_list)[ind_list])
mask_path_list = list(np.array(mask_path_list)[ind_list])

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)


# PREPARE IMAGE AND MASK LISTS

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list1 = image_path_list[valid_ind:]
train_label_path_list1 = mask_path_list[valid_ind:]


train_input_path_list=train_input_path_list1

train_label_path_list=train_label_path_list1
# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size

# CALL MODEL

model = UNet(3, 2,True)


#train_and_valid=train_input_path_list+valid_input_path_list
#train_and_valid_labeld=train_label_path_list+valid_label_path_list









criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    
    
    # IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()

val_losses=[]
train_losses=[]
# TRAINING THE NEURAL NETWORK


scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1, patience=1, verbose=True)

   
   
for epoch in range(epochs):
    print("----------","Epoch : ",epoch,"-----------------")
    correct_train=0
    total_train=0
    losses=[]

    running_loss = 0
    for ind in tqdm.tqdm(range(steps_per_epoch)):

        #torch.save(model.state_dict(), "../save.pth.tar")
        #print("saved")
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        #print("parth list:" ,train_input_path_list)
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
       # print("batch list ",batch_label_path_list)
        batch_input = tensorize_image(batch_input_path_list, input_shape)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)
        #print(batch_label)


        optimizer.zero_grad()


        outputs = model(batch_input)


        loss = criterion(outputs, batch_label)
        loss.backward()# Calculates the gradient, how much each parameter needs to be updated
       # print(loss.data)
        optimizer.step()
        
        running_loss += loss.item()



        predicted2= (outputs>0.5).float()
        total_train += batch_label.nelement()
        correct_train+= (predicted2== batch_label).float().sum().item()
        losses.append(loss.item())

        if ind == steps_per_epoch-1:
            model.eval()
            print()
            print('training loss on epoch {}: {}'.format(epoch, running_loss/steps_per_epoch))
            print('Accuracy for train   %', ( 100.0 * correct_train / total_train))
            total_val=0
            corret_val=0

            val_loss = 0
            train_losses.append(running_loss/steps_per_epoch)
            for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                outputs = model(batch_input)

                # valid accuracy 
                predictval=(outputs>0.5).float()
                total_val += batch_label.nelement()
                corret_val+= (predictval== batch_label).float().sum().item()

                val_loss += loss.item()
                

                val_losses.append( val_loss/len(valid_input_path_list))
        model.train()
                
                
    scheduler.step(sum(val_losses)/len(val_losses))       
    print('validation loss on epoch '.format(epoch, val_loss/len(valid_input_path_list)))
    print('Accuracy for valid  %', ( 100.0 * corret_val / total_val))
torch.save(model.state_dict(),"../models/model.pth.tar")
print("model saved")



"""


def draw_graph(val_losses,train_losses,epochs):
   
    norm_validation = [float(i)/sum(val_losses) for i in val_losses]
    norm_train = [float(i)/sum(train_losses) for i in train_losses]
    
    epoch_numbers=list(range(1,epochs+1,1))
    plt.figure(figsize=(12,6))
    plt.subplot(2, 2, 1)
    plt.plot(epoch_numbers,norm_validation,color="red") 
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Train losses')
    plt.subplot(2, 2, 2)
    plt.plot(epoch_numbers,norm_train,color="blue")
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Validation losses')
    plt.subplot(2, 1, 2)
    plt.plot(epoch_numbers,norm_validation, 'r-',color="red")
    plt.plot(epoch_numbers,norm_train, 'r-',color="blue")
    plt.legend(['w=1','w=2'])
    plt.title('Train and Validation Losses')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    
   # plt.savefig(fname="graf.png")
    plt.show()
draw_graph(val_losses,train_losses,epochs)
"""

# for predict

"""
test_input_path_list = image_path_list
test_label_path_list=mask_path_list


model = UNet(3, 2,True)
device = torch.device('cpu')

model.load_state_dict(torch.load("../models/remodel9.pth", map_location=device))
model.eval()



correct_predict=0
wrong_predict = 0
total_predict=0
for i in tqdm.tqdm(range(len(test_input_path_list))):
    batch_test = test_input_path_list[i:i+1]
    test_input = tensorize_image(batch_test, input_shape, cuda)
    outs = model(test_input)
    out=torch.argmax(outs,axis=1)
    out_cpu = out.cpu()
    outputs_list=out_cpu.detach().numpy()
    mask=np.squeeze(outputs_list,axis=0)
    
    batch_label= test_label_path_list[i:i+1]
    test_label= tensorize_mask(batch_label,input_shape,n_classes,cuda)
    out_test_acc= (outs>0.5).float()
    correct_predict+= (out_test_acc==test_label).float().sum().item()
    wrong_predict += (out_test_acc!=test_label).float().sum().item()
    
    
        
        
    img=cv2.imread(batch_test[0])
    #mg=cv2.resize(img,(224,224))
    mask=cv2.resize(mask.astype(np.uint8),(1920,1208))
    mask_ind   = mask == 1
    cpy_img  = img.copy()
    img[mask==1 ,:] = (255, 0, 125)
    opac_image=(img/2+cpy_img/2).astype(np.uint8)
    predict_name=batch_test[0]
    predict_path=predict_name.replace('images', 'predict')
    cv2.imwrite(predict_path,opac_image.astype(np.uint8))
total_predict = correct_predict + wrong_predict
#print(correct_predict, " ", total_predict)
acc_test=100.0* correct_predict / total_predict
print("Test accuracy: ",acc_test)




"""






