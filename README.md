# Logo detection with YoloV5

## Run
```python3
import tqdm
import cv2
from logo import LogoDetector
from config import MODEL_PATH
from scene_detection.detector import Detector

LD = LogoDetector(model_path=MODEL_PATH)
detector = Detector()
```

### Full video process
```python3
cap = cv2.VideoCapture('resources/test.mp4')
ret, frame = cap.read()
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('markup.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

bar = tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

while ret:
     try:
         _, i = LD.process(img0=frame, conf_threshold=0.3, verbose=False, draw=True)
         video.write(i)
     except TypeError:
         pass
     except KeyboardInterrupt:
         break
     ret, frame = cap.read()
     bar.update()
     bar.refresh()

video.release()
cv2.destroyAllWindows()
```

### Scene split video process
```python3
frames = detector.get_scene_frames(video_file_path='resources/test.mp4')
for n, frame in tqdm.tqdm(enumerate(frames)):
    logos, image = LD.process(img0=frame, conf_threshold=0.1, iou_threshold=0.1, verbose=False, draw=True)
    if logos is not None:
        cv2.imwrite('resources/test_results/{0}.png'.format(n), image)
```
