#encoding=gbk
import os.path

import onnx
import onnxruntime as ort
import numpy as np
import cv2
from tqdm import  tqdm
from imutils import paths
#路径配置
onnx_path=r'./onnx/best-ort.onnx'
imgspath=r'./onnx/imgs/'
w,h=640,640

if not os.path.exists('./onnx/results'):
    os.makedirs('./onnx/results')

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
#        2:'elephant',
#        3:'bird'}
label={0:'BOX'}


imgpaths=list(paths.list_images(imgspath))

#onnx模型载入
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])

for pic_path in tqdm(imgpaths):
    basename=os.path.basename(pic_path)
    img=cv2.imread(pic_path)
    H,W=img.shape[0],img.shape[1]
    h_ratio=H/h
    w_ratio=W/w
    imgbak=img.copy()
    img=cv2.resize(img,(w,h)).astype(np.float32)
    img=np.array([np.transpose(img,(2,0,1))])


    #模型推理
    out = session.run(None,input_feed = { 'input' : img })

    obj_nums=int(out[0][0][0])
    for i in range(obj_nums):
        id=out[3][0][i]
        score=out[2][0][i]
        x1,y1,x2,y2=out[1][0][i][0],out[1][0][i][1],out[1][0][i][2],out[1][0][i][3]
        x1=int(w_ratio*x1)
        x2 = int(w_ratio * x2)
        y1 = int(h_ratio * y1)
        y2 = int(h_ratio * y2)

        cv2.rectangle(imgbak,(x1,y1),(x2,y2),palette[int(id)],3)
        cv2.putText(imgbak, '{}:{:.2f}'.format(label[int(id)], float(score)), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, palette[int(id)], 1)
    cv2.imwrite(os.path.join('./onnx/results',basename), imgbak)








