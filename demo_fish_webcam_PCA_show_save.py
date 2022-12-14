# not done
# edit by huangyuan 2022.04.26
# 
# read folder imgs using solov2 and PCA,show on screen and save files
# no serial inside



from queue import Empty
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
#checkpoint_file = '../work_dirs/solov2_release_r50_fpn_8gpu_3x_changed_202201_trained/epoch_120.pth'
checkpoint_file = '../checkpoints/epoch_50.pth'
# build the model from a config file and a checkpoint file
model = inference.init_detector(config_file, checkpoint_file, device='cuda:0')

output_folder = 'output/webcam_test/'
t = time.localtime()
output_folder += time.strftime("%Y%m%d_%H%M_", t)
"""
input_folder= "input/20fish_ntust/"
output_folder = 'output/20fish_ntust_test/'
input_folder_names = os.listdir(input_folder)
img_names = []
for name in input_folder_names:
    if name[-4:]=='.jpg':
        img_names.append(name[:-4])
print("from ",input_folder)
print(img_names," is img_names")
"""
camera = cv2.VideoCapture(0)

total_weight = 0
count = 0
# test a single image
while True:
    
    ch = cv2.waitKey(1)
    if ch == 27 or ch == ord('q') or ch == ord('Q'):
        break
    

    ret_val, img = camera.read()

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
    #print(weight_arr)

    if count == 0:
        cv2.imshow("dual",img)

    try:
        #print("shape::",weight_arr.shape[0])
        if not(weight_arr.shape[0]):
            continue
        if weight_arr==None:#temporty hide error in PCA_import # TODO
            continue
    except:
        continue

    #print(weight_arr.shape)
    #if(weight_arr.shape):
    #    continue

    do_index = np.argmax(weight_arr)
    std_weight_arr = np.array([700,600,500])
    #print(do_index,weight_arr)

    if   weight_arr[do_index]>std_weight_arr[0]: gate = 1 #700
    elif weight_arr[do_index]>std_weight_arr[1]: gate = 2 #600
    elif weight_arr[do_index]>std_weight_arr[2]: gate = 3 #500
    else:                                        gate = 4

    if   gate == 1:category_message = "{}????????????".format(std_weight_arr[0])
    elif gate == 2:category_message = "{}?????????{}??????".format(std_weight_arr[1],std_weight_arr[0])
    elif gate == 3:category_message = "{}?????????{}??????".format(std_weight_arr[2],std_weight_arr[1])
    elif gate == 4:category_message = "{}????????????".format(std_weight_arr[2])


    total_weight += weight_arr[do_index]   

    edited_page = inference.drawChinese(edited_page,"??????:{:.1f}??????".format(length_arr[do_index]),(10,50))#350
    edited_page = inference.drawChinese(edited_page,"??????:{:.1f}??????".format(height_arr[do_index]),(10,100))#400
    edited_page = inference.drawChinese(edited_page,"??????:{:.1f}??????".format(weight_arr[do_index]),(10,150))
    edited_page = inference.drawChinese(edited_page,"??????:{}".format(category_message),(10,200))
    edited_page = inference.drawChinese(edited_page,"??????:{}".format(count+1),(10,250))
    edited_page = inference.drawChinese(edited_page,"??????:{:.2f}??????".format(total_weight/1000),(10,300))
    count += 1
    
    cv2.namedWindow("dual", cv2.WINDOW_AUTOSIZE)
    if cv2.getWindowProperty("dual", 0) >= 0 :
        ##dual_image = np.hstack((img, edited_page))
        dual_image = np.hstack((img_out, edited_page))
        #dual_image = np.vstack((img, edited_page))

        dheight, dwidth, layers = dual_image.shape
        ddsize = (int(dwidth*0.7),int(height*0.7))
        #print(ddsize)
        dual_image = cv2.resize(dual_image,ddsize)

        cv2.imshow("dual", dual_image)
        cv2.imwrite(output_folder+str(count)+'.jpg',dual_image)
    """
    key = cv2.waitKey(0)
    #cv2.destroyAllWindows()
    if key == 113:#q
        break
    """
    
cv2.destroyAllWindows()