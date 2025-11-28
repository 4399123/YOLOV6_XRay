#encoding=gbk
import os.path

import onnx
import onnxruntime as ort
import numpy as np
import cv2
from tqdm import  tqdm
from imutils import paths
import random
#路径配置
onnx_path=r'./onnx/best-ort.onnx'
imgspath=r'C:\G\Baofeng\proj\3_det_seg\all_imgs\imgsOK2'
# imgspath=r'./onnx/imgs'
w,h=640,640
cut_x,cut_y=288,288

delta_x,delta_y=20,-20

if not os.path.exists('./onnx/results'):
    os.makedirs('./onnx/results')

if not os.path.exists('./onnx/results_cut'):
    os.makedirs('./onnx/results_cut')

palette={0:(0,255,0),
    1:(0,0,255),
    2:(255,0,0),
    3:(255,255,0),
    4:(255,0,255),
    5:(171,130,255),
    6:(155,211,255),
    7:(0,255,255)}


# label={0:'cat',
#        1:'dog',
#        2:'eagle',
#        3:'elephant'}
label={0:'TuAn',
       1:'LV'}


imgpaths=list(paths.list_images(imgspath))
random.shuffle(imgpaths)

#onnx模型载入
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path,providers=['CUDAExecutionProvider'])

for pic_path in tqdm(imgpaths):
    basename=os.path.basename(pic_path)
    name=basename.split('.')[0]
    img=cv2.imread(pic_path)
    H,W=img.shape[0],img.shape[1]
    h_ratio=H/h
    w_ratio=W/w
    imgbak=img.copy()
    imgbak2=img.copy()
    # img=cv2.resize(img,(w,h)).astype(np.float32)
    img = cv2.resize(img, (w, h))
    img=np.array([np.transpose(img,(2,0,1))])


    #模型推理
    out = session.run(None,input_feed = { 'input' : img })

    obj_nums=int(out[0][0][0])
    lv_xy=[]
    for i in range(obj_nums):
        id=out[3][0][i]

        if(id==0): continue

        score=out[2][0][i]
        x1,y1,x2,y2=out[1][0][i][0],out[1][0][i][1],out[1][0][i][2],out[1][0][i][3]
        x1=int(w_ratio*x1)
        x2 = int(w_ratio * x2)
        y1 = int(h_ratio * y1)
        y2 = int(h_ratio * y2)


        xx=x1-delta_x
        yy=y1-delta_y

        if(xx<0): xx=0
        if(yy<0): yy=0

        lv_xy.append([xx,yy])


        cv2.rectangle(imgbak,(x1,y1),(x2,y2),palette[int(id)],4)

        cv2.rectangle(imgbak, (xx, yy), (xx+cut_x, yy+cut_y),(155,211,255), 4)
        cv2.putText(imgbak, '{}:{:.6f}'.format(label[int(id)], float(score)), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, palette[int(id)], 2)
    cv2.imwrite(os.path.join('./onnx/results',name+'.jpg'), imgbak)

    for k in range(len(lv_xy)):
        cutimg=imgbak2[lv_xy[k][1]:lv_xy[k][1]+cut_y,lv_xy[k][0]:lv_xy[k][0]+cut_x]
        cv2.imwrite(os.path.join('./onnx/results_cut', name + '_{}.png'.format(int(k))), cutimg)








