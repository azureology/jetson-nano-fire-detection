import time
import os
import cv2
import torch

from nanodet.util import cfg, load_config, Logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline

cam_id = 0
config_path = 'fire/nanodet_fire.yml'
model_path = 'fire/model_fire.pth'

class Predictor(object):
    def __init__(self, cfg, model_path, logger, device='cuda:0'):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {}
        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        meta = dict(img_info=img_info,raw_img=img,img=img)
        meta = self.pipeline(meta, self.cfg.data.val.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        self.model.head.show_result(meta['raw_img'], dets, class_names, score_thres=score_thres, show=True)
        print('viz time: {:.3f}s'.format(time.time()-time1))

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    load_config(cfg, config_path)
    logger = Logger(-1, use_tensorboard=False)
    predictor = Predictor(cfg, model_path, logger, device='cuda:0')
    logger.log('Press "Esc" to exit.')
    cap = cv2.VideoCapture(cam_id) #str for video
    while True:
        ret_val, frame = cap.read()
        if ret_val == False:
            continue #skip if capture fail
        meta, res = predictor.inference(frame)
        predictor.visualize(res, meta, cfg.class_names, 0.35)
        ch = cv2.waitKey(1)
        if ch == 27:
            break