# not done
# edit by huangyuan 2022.04.18
# 
# read folder imgs using solov2 and PCA,show on screen and save files
# no serial inside



from tkinter import W
from mmdet.apis import inference_PCA as inference
#from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import warnings
warnings.filterwarnings("ignore")
import cv2
import time
import os
import numpy as np
  

config_file = '../configs/solov2/solov2_r50_fpn_8gpu_3x_change.py'
checkpoint_file = '../work_dirs/solov2_release_r50_fpn_8gpu_3x_changed_202201_trained/epoch_120.pth'

# build the model from a config file and a checkpoint file
model = inference.init_detector(config_file, checkpoint_file, device='cuda:0')

input_folder= "input/20fish_ntust/"
output_folder = 'output/20fish_ntust_test/'
input_folder_names = os.listdir(input_folder)
img_names = []
for name in input_folder_names:
    if name[-4:]=='.jpg':
        img_names.append(name[:-4])
print("from ",input_folder)
print(img_names," is img_names")

total_weight = 0
# test a single image
for i in range(len(img_names)):
    filename=input_folder + img_names[i]
    #reading each files
    img = cv2.imread(filename+'.jpg')
    height, width, layers = img.shape
    size = (width,height)

    shape = (height,350,3)
    edited_page = np.zeros(shape,np.uint8)
    #cv2.imshow("edited", edited_page)
    edited_page.fill(255)  

    #img = '4622shot.jpg'
    result = inference.inference_detector(model, img)

    img_out,length_arr,height_arr,weight_arr = inference.show_result_PCA(img, result, model.CLASSES, score_thr=0.25)

    #max_weight = max(weight_arr)
    #do_index = weight_arr.index(max_weight)
    do_index = np.argmax(weight_arr)
    #print(do_index,weight_arr)

    if   weight_arr[do_index]>700: gate = 1
    elif weight_arr[do_index]>600: gate = 2
    elif weight_arr[do_index]>500: gate = 3
    else:                          gate = 4

    total_weight += weight_arr[do_index]   

    edited_page = inference.drawChinese(edited_page,"體長:{:.1f}公分".format(length_arr[do_index]),(10,350))
    edited_page = inference.drawChinese(edited_page,"體高:{:.1f}公分".format(height_arr[do_index]),(10,400))
    edited_page = inference.drawChinese(edited_page,"重量:{:.1f}公克".format(weight_arr[do_index]),(10,450))
    edited_page = inference.drawChinese(edited_page,"分類:{}".format(gate),(10,500))
    edited_page = inference.drawChinese(edited_page,"總數:{}".format(i+1),(10,550))
    edited_page = inference.drawChinese(edited_page,"總重:{:.2f}公斤".format(total_weight/1000),(10,600))
    # count += 1
    
    cv2.namedWindow("dual", cv2.WINDOW_AUTOSIZE)
    if cv2.getWindowProperty("dual", 0) >= 0 :
        dual_image = np.hstack((img, edited_page))
        #dual_image = np.vstack((img, edited_page))

        dheight, dwidth, layers = dual_image.shape
        ddsize = (int(dwidth*0.7),int(height*0.7))
        #print(ddsize)
        dual_image = cv2.resize(dual_image,ddsize)

        cv2.imshow("dual", dual_image)
    key = cv2.waitKey(0)
    #cv2.destroyAllWindows()
    if key == 113:#q
        break
    
cv2.destroyAllWindows()