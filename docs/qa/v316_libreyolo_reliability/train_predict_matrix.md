# v3.16 LibreYOLO train→reload→eval-vs-predict matrix

epochs=25 device=cpu

| Model | verdict | eval mAP50 | predict@0.25 | predict@0.05 |
|---|---|--:|--:|--:|
| libreyolo-yolox-s | PREDICT_OK | 1.0 | 1 | 1 |
| libreyolo-yolov9-s | PREDICT_OK | 0.9504950495049505 | 1 | 2 |
| libreyolo-rtdetr-r50 | PREDICT_OK | 0.43552569542668546 | 2 | 38 |
| libreyolo-dfine-n | CRASHED | None | None | None |
