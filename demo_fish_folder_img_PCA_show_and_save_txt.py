# not done
# edit by huangyuan 2022.04.18
# 
# read folder imgs using solov2 and PCA,show on screen and save files
# no serial inside



from tkinter import W
from mmdet.apis import inference_PCA as inference
from mmdet.apis import mysql_import as mysql
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

#input_folder= "input/20fish_ntust/"
#output_folder = 'output/20fish_ntust_test/'
input_folder= "input/penghu/test02index/"
output_folder = 'output/penghu/test02index/'

input_folder= "input/penghu/test03cho/"
output_folder = 'output/penghu/test03cho/default/'


#output_txt_path = output_folder+'20demotxt.txt'
output_txt_path = output_folder+'penghutest_demotxt.txt'
input_folder_names = os.listdir(input_folder)
img_names = []
for name in input_folder_names:
    if name[-4:]=='.jpg':
        img_names.append(name[:-4])
print("from ",input_folder)
print(img_names," is img_names")
print("total {} images".format(len(img_names)) )

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
    #std_weight_arr = np.array([300,250,200])
    std_weight_arr = np.array([500,400,300])
    std_weight_arr = np.array([700,600,500])
    print(img_names[i],do_index,weight_arr)

    if   weight_arr[do_index]>std_weight_arr[0]: 
        gate = 1 #700
        category_message_ch = "{}公克以上".format(std_weight_arr[0])
        category_message_en = "above_{}g".format(std_weight_arr[0])
    elif weight_arr[do_index]>std_weight_arr[1]: 
        gate = 2 #600
        category_message_ch = "{}公克至{}公克".format(std_weight_arr[1],std_weight_arr[0])
        category_message_en = "{}g_to_{}g".format(std_weight_arr[1],std_weight_arr[0])
    elif weight_arr[do_index]>std_weight_arr[2]: 
        gate = 3 #500
        category_message_ch = "{}公克至{}公克".format(std_weight_arr[2],std_weight_arr[1])
        category_message_en = "{}g_to_{}g".format(std_weight_arr[2],std_weight_arr[1])
    else:                                        
        gate = 4
        category_message_ch = "{}公克以下".format(std_weight_arr[2])
        category_message_en = "under_{}g".format(std_weight_arr[2])

    
    total_weight += weight_arr[do_index]   

    edited_page = inference.drawChinese(edited_page,"體長:{:.1f}公分".format(length_arr[do_index]),(10,350))
    edited_page = inference.drawChinese(edited_page,"體高:{:.1f}公分".format(height_arr[do_index]),(10,400))
    edited_page = inference.drawChinese(edited_page,"重量:{:.1f}公克".format(weight_arr[do_index]),(10,450))
    edited_page = inference.drawChinese(edited_page,"分級:{}".format(category_message_ch),(10,500))
    edited_page = inference.drawChinese(edited_page,"總數:{}".format(i+1),(10,550))
    edited_page = inference.drawChinese(edited_page,"總重:{:.2f}公斤".format(total_weight/1000),(10,600))
    # count += 1
    try:
        with open(output_txt_path,'a')as f:
            f.write('{:.1f},{:.1f},{:.1f},{},{},{:.2f}\n'.format(length_arr[do_index], height_arr[do_index], weight_arr[do_index], category_message_en, i+1, total_weight/1000))
    except:
        with open(output_txt_path,'w')as f:
            f.write('{:.1f},{:.1f},{:.1f},{},{},{:.2f}\n'.format(length_arr[do_index], height_arr[do_index], weight_arr[do_index], category_message_en, i+1, total_weight/1000))

    mysql.sendToDB_message(length_arr[do_index], height_arr[do_index], weight_arr[do_index], i+1, category_message_en, total_weight/1000)
    ####mysql_not_needed_when_testing
    print("mysql send {} finish".format(i+1))
    
    cv2.namedWindow("dual", cv2.WINDOW_AUTOSIZE)
    if cv2.getWindowProperty("dual", 0) >= 0 :
        dual_image = np.hstack((img_out, edited_page))
        #dual_image = np.hstack((img, edited_page))
        #dual_image = np.vstack((img, edited_page))

        dheight, dwidth, layers = dual_image.shape
        ddsize = (int(dwidth*0.7),int(height*0.7))
        #print(ddsize)
        dual_image = cv2.resize(dual_image,ddsize)

        cv2.imshow("dual", dual_image)
        #cv2.imwrite(output_folder+str(i)+'.jpg',dual_image)
        
        #cv2.imwrite('{}{:04d}.jpg'.format(output_folder,i+1),dual_image)
        cv2.imwrite('{}{}.jpg'.format(output_folder,img_names[i]),dual_image)
    """
    key = cv2.waitKey(0)
    #cv2.destroyAllWindows()
    if key == 113:#q
        break
    """
    
cv2.destroyAllWindows()
