#encoding=gbk
import os.path

import onnx
import onnxruntime as ort
import numpy as np
import cv2
from tqdm import  tqdm
from imutils import paths
import datetime
#路径配置
onnx_path=r'./onnx/best-ort.onnx'
imgspath=r'C:\G\Baofeng\proj\3_det_seg\all_imgs\imgs1114'
# imgspath=r'./onnx/imgs'
w,h=640,640
cut_x,cut_y=896,896

if not os.path.exists('./onnx/results'):
    os.makedirs('./onnx/results')
if not os.path.exists('./onnx/cutimgs'):
    os.makedirs('./onnx/cutimgs')

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
label={0:'TuAn'}


imgpaths=list(paths.list_images(imgspath))

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
    # img=cv2.resize(img,(w,h)).astype(np.float32)
    img = cv2.resize(img, (w, h))
    img=np.array([np.transpose(img,(2,0,1))])


    #模型推理
    out = session.run(None,input_feed = { 'input' : img })

    obj_nums=int(out[0][0][0])
    n=0
    for i in range(obj_nums):
        id=out[3][0][i]
        score=out[2][0][i]
        x1,y1,x2,y2=out[1][0][i][0],out[1][0][i][1],out[1][0][i][2],out[1][0][i][3]
        x1=int(w_ratio*x1)
        x2 = int(w_ratio * x2)
        y1 = int(h_ratio * y1)
        y2 = int(h_ratio * y2)

        # if((x2-x1)<200 or(y2-y1)<200):continue

        center_x=int((x1+x2)/2)
        center_y=int((y1+y2)/2)
        xx1=center_x-cut_x//2
        yy1=center_y-cut_y//2
        xx2=center_x+cut_x//2
        yy2=center_y+cut_y//2

        # xx1=max(0,xx1)
        # yy1=max(0,yy1)
        # xx2=min(W,xx2)
        # yy2=min(H,yy2)

        if (xx1<0):
            xx1=0
            xx2=cut_x
        elif(xx2>W):
            xx1=W-cut_x
            xx2=W
        else:pass

        if (yy1 < 0):
            yy1 = 0
            yy2 = cut_y
        elif (yy2 > H):
            yy1 = H - cut_y
            yy2 = H
        else: pass


        cut_img=imgbak[yy1:yy2,xx1:xx2]
        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.imwrite('./onnx/cutimgs/{}_{}_{}.png'.format(name,n,current_time),cut_img)
        n+=1

    #     ###保存大图
    #     cv2.rectangle(imgbak, (xx1, yy1), (xx2, yy2), palette[int(id)], 4)
    #     cv2.putText(imgbak, '{}:{:.6f}'.format(label[int(id)], float(score)), (xx1, yy1 - 5),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 2, palette[int(id)], 2)
    # cv2.imwrite(os.path.join('./onnx/results', basename.replace('.png', '.jpg')), imgbak)










