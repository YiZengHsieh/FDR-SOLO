from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv


config_file = '../configs/solov2/solov2_r50_fpn_8gpu_3x_change.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# heckpoint_file = '../checkpoints/epoch_120.pth'
#checkpoint_file = '../work_dirs/solov2_release_r50_fpn_8gpu_3x_changed/epoch_120.pth'
checkpoint_file = '../checkpoints/epoch_50.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = '0031.jpg'
result = inference_detector(model, img)

show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file="demo_outfish.jpg")
