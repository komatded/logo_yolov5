from image_preprocess import preprocess_image
from prediction_process import *
from config import *
import torch
import cv2


class LogoDetector:

    def __init__(self, model_path):
        self.device = torch.device('cpu')
        self.model = torch.load(model_path, map_location=self.device)['model'].float()
        self.model.to(self.device).eval()

    def _preprocess(self, img0):
        img = preprocess_image(image=img0.copy(), image_size=IMAGE_SIZE)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    @staticmethod
    def _draw_result(img0, prediction):
        draw = img0.copy()
        if prediction is not None:
            for x0, y0, x1, y1, conf, _ in prediction:
                draw = cv2.rectangle(draw, (x0, y0), (x1, y1), (255, 0, 0), 3)
                draw = cv2.putText(draw, str(round(float(conf), 2)), (x0, y1),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return draw

    def process(self, img0=None, image_path=None, conf_threshold=0.4, iou_threshold=0.5, verbose=True, draw=False):
        if img0 is None:
            img0 = cv2.imread(image_path)
        start = time.time()
        img = self._preprocess(img0)
        t1 = time.time()
        prediction = self.model(img, augment=False)[0]
        t2 = time.time()
        prediction = non_max_suppression(prediction, conf_threshold, iou_threshold)[0]
        if prediction is not None:
            prediction[:, :4] = scale_coords(img.shape[2:], prediction[:, :4], img0.shape).round()
        t3 = time.time()
        t1, t2, t3 = t1 - start, t2 - t1, t3 - t2
        if verbose:
            logger.info('Time: image preprocess: {0}s, prediction: {1}s, prediction process: {2}s'.format(round(t1, 3),
                                                                                                          round(t2, 3),
                                                                                                          round(t3, 3)))
        if draw:
            return prediction, self._draw_result(img0, prediction)
        return prediction, None
