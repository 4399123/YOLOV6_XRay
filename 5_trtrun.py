#encoding=gbk
import tensorrt as trt
import numpy as np
import os
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from imutils import paths
from tqdm import tqdm


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtModel:
    def __init__(self, engine_path, max_batch_size=1):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        # 移除 self.dtype = np.float32，改为动态分配
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            # 1. 获取该绑定的维度
            dims = self.engine.get_binding_shape(binding)

            # 处理动态 Batch Size (如果第一维是 -1)
            if dims[0] == -1:
                dims[0] = self.max_batch_size

            # 计算总元素数量
            size = trt.volume(dims)

            # 2. 【关键修改】动态获取该绑定的数据类型 (float32 或 uint8)
            trt_dtype = self.engine.get_binding_dtype(binding)

            # 将 TensorRT 类型映射到 NumPy 类型
            if trt_dtype == trt.float32:
                dtype = np.float32
            elif trt_dtype == trt.uint8:
                dtype = np.uint8
            elif trt_dtype == trt.int8:
                dtype = np.int8
            else:
                dtype = np.float32  # 默认 fallback

            # 3. 按照正确的类型分配内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def __call__(self, x: np.ndarray, batch_size=1):
        # 【关键修改】不再强制转为 float32 (self.dtype)
        # 而是转换成输入 buffer 实际期望的类型
        input_needed_dtype = self.inputs[0].host.dtype

        if x.dtype != input_needed_dtype:
            x = x.astype(input_needed_dtype)

        np.copyto(self.inputs[0].host, x.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        # 这里的 reshape 逻辑保持不变
        origin_inputshape = list(self.engine.get_binding_shape(0))
        origin_inputshape[0] = batch_size
        self.context.set_binding_shape(0, tuple(origin_inputshape))

        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()

        # 输出通常还是 float32，直接 reshape 即可
        return [out.host.reshape(batch_size, -1) for out in self.outputs]


if __name__ == "__main__":

    w, h = 640, 640
    path=r'./onnx/imgs/'
    trt_engine_path = r'./onnx/best-trt.engine'

    if not os.path.exists('./onnx/results'):
        os.makedirs('./onnx/results')

    palette = {0: (0, 255, 0),
               1: (0, 0, 255),
               2: (255, 0, 0),
               3: (255, 255, 0),
               4: (255, 0, 255),
               5: (171, 130, 255),
               6: (155, 211, 255),
               7: (0, 255, 255)}

    label = {0: 'BOX'}


    model = TrtModel(trt_engine_path)

    pic_paths = list(paths.list_images(path))
    for pic_path in tqdm(pic_paths):
        basename=os.path.basename(pic_path)
        img = cv2.imread(pic_path)
        imgbak=img.copy()
        H, W = img.shape[0], img.shape[1]
        h_ratio = H / h
        w_ratio = W / w
        img=cv2.resize(img,(w,h))
        img = np.array([np.transpose(img, (2, 0, 1))])

        out = model(img, 1)

        obj_nums=int(out[2][0][0])

        for i in range(obj_nums):
            id = out[3][0][i]          #锟斤拷锟絀D
            score = out[1][0][i]       #锟斤拷锟矫凤拷
            x1, y1, x2, y2 = out[0][0][i*4+0], out[0][0][i*4+1], out[0][0][i*4+2], out[0][0][i*4+3]     #锟斤拷锟斤拷锟斤拷辖呛锟斤拷锟斤拷陆锟斤拷锟斤拷锟?

            x1 = int(w_ratio * x1)
            x2 = int(w_ratio * x2)
            y1 = int(h_ratio * y1)
            y2 = int(h_ratio * y2)

            cv2.rectangle(imgbak, (x1, y1), (x2, y2), palette[int(id)], 3)
            cv2.putText(imgbak, '{}:{:.2f}'.format(label[int(id)], float(score)), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, palette[int(id)], 1)
        cv2.imwrite(os.path.join('./onnx/results',basename),imgbak)





