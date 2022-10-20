from mmdet.apis import inference_PCA as inference
#from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import warnings
warnings.filterwarnings("ignore")
import cv2
import serial, time
  
ser = serial.Serial()
port = input('enter the COM name there(ex: COM3 ): ')
ser.port = port
#115200,N,8,1
ser.baudrate = 9600
ser.bytesize = serial.EIGHTBITS #number of bits per bytes
ser.parity = serial.PARITY_NONE #set parity check
ser.stopbits = serial.STOPBITS_ONE #number of stop bits
  
ser.timeout = 0.5          #non-block read 0.5s
ser.writeTimeout = 0.5     #timeout for write 0.5s
ser.xonxoff = False    #disable software flow control
ser.rtscts = False     #disable hardware (RTS/CTS) flow control
ser.dsrdtr = False     #disable hardware (DSR/DTR) flow control
try: 
    ser.open()
except Exception as ex:
    print ("open serial port error " + str(ex))
    exit()

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
        ser.close()
        exit()    
    #img = '4622shot.jpg'
    result = inference.inference_detector(model, img)

    img_out,weight_array = inference.show_result_PCA(img, result, model.CLASSES, score_thr=0.25)
    
    for weight in weight_array:#700-600-500
        if weight>400:
            gate = 1
        elif weight>350:
            gate = 2
        elif weight>300:
            gate = 3
        else:
            gate = 4    

    if ser.isOpen():
        try:
            ser.flushInput() #flush input buffer
            ser.flushOutput() #flush output buffer
            ser.write([int(gate)])
            print("ser.write gate: ",gate)            
        except Exception as e1:
            print ("communicating error " + str(e1))
    else:
        print ("open serial port error")

            
    cv2.imshow('out',img_out)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
