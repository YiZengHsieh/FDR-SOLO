from mmdet.apis import inference_PCA as inference
#from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import warnings
warnings.filterwarnings("ignore")
import cv2


config_file = '../configs/solov2/solov2_r50_fpn_8gpu_3x_change.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# heckpoint_file = '../checkpoints/epoch_120.pth'
checkpoint_file = '../work_dirs/solov2_release_r50_fpn_8gpu_3x_changed_202201_trained/epoch_120.pth'

# build the model from a config file and a checkpoint file
model = inference.init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
while(1):
    img = input('enter the file name there: ')
    if(img=='q'):
        exit()    
    #img = '4622shot.jpg'
    result = inference.inference_detector(model, img)

    img_out,weights = inference.show_result_PCA(img, result, model.CLASSES, score_thr=0.25)
    cv2.imshow('out',img_out)
    cv2.waitKey()#500
    cv2.destroyAllWindows()
    print(weights)
