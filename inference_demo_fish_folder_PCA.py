# demo folder control concept from mask11
# edited by huangyuan at 2022.03.04
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
from mmdet.apis import inference_PCA as inference
import mmcv
import os
import time


config_file = '../configs/solov2/solov2_r50_fpn_8gpu_3x_change.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# heckpoint_file = '../checkpoints/epoch_120.pth'
##checkpoint_file = '../work_dirs/solov2_release_r50_fpn_8gpu_3x_changed/epoch_120.pth'
#checkpoint_file = '../work_dirs/solov2_release_r50_fpn_8gpu_3x_changed_202201_trained/epoch_50.pth'
checkpoint_file = '../checkpoints/epoch_50.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
#img = '0024.jpg'
#result = inference_detector(model, img)
#show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file="demo_outfish.jpg")

#input_folder = "40fishes/"
#input_folder = "13fish/"
input_folder = "input/20fish_ntust/"
#input_folder = "input/11770_cap/"
output_folder = "output/20fish_ntust_demo/"
#output_folder = "output/11770cap_demo/"
input_folder_names = os.listdir(input_folder)
img_names = []
for name in input_folder_names:
    if name[-4:]=='.jpg':
        img_names.append(name[:-4])
print("from ",input_folder)
print(img_names," is img_names")

for i in range(len(img_names)):
    img_name = img_names[i]
    img = input_folder+img_name+'.jpg'
    a = time.monotonic()
    result = inference_detector(model, img)
    b = time.monotonic()
    #show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file=output_folder+"demo_"+img_name+".jpg")
    inference.show_result_PCA(img, result, model.CLASSES, score_thr=0.25, out_file=output_folder+"demo_"+img_name+".jpg")
    print("detect time = {:.2f}".format(b-a) )

