import os
from .bounding_box_filter import BoundingBoxFilter

class PatchProcessor:
    def __init__(self, model, patch_path):
        self.model = model
        self.patch_path = patch_path

    def process(self):
        filename = os.path.basename(self.patch_path)
        # filename like "patch_{y}_{x}.png"
        parts = filename.replace(".png", "").split('_')[1:]
        offset = [int(p) for p in parts]

        results = self.model(self.patch_path)
        wh_boxes = results[0].boxes.xywh.cpu().numpy().tolist()
        # convert to xyxy in full-image coords
        bboxs = [
            [
                (x - w/2) + offset[1],
                (y - h/2) + offset[0],
                (x + w/2) + offset[1],
                (y + h/2) + offset[0]
            ]
            for x, y, w, h in wh_boxes
        ]
        labels = results[0].boxes.cls.cpu().numpy().tolist()

        pl, pt = offset[1], offset[0]
        pr, pb = pl + 640, pt + 640
        det = [o // 320 % 2 for o in offset]
        if det == [0,0]:
            return BoundingBoxFilter.delete_edge(bboxs, labels, pl, pt, pr, pb)
        if det == [0,1]:
            return BoundingBoxFilter.keep_cline(bboxs, labels, pl, pt, pr, pb)
        if det == [1,0]:
            return BoundingBoxFilter.keep_rline(bboxs, labels, pl, pt, pr, pb)
        return BoundingBoxFilter.keep_center(bboxs, labels, pl, pt, pr, pb)
    
    def one_img_process(self):
        filename = os.path.basename(self.patch_path)
        # filename like "patch_{y}_{x}.png"
        parts = filename.replace(".png", "").split('_')[1:]
        offset = [int(p) for p in parts]

        results = self.model(self.patch_path)
        wh_boxes = results[0].boxes.xywh.cpu().numpy().tolist()
        # convert to xyxy in full-image coords
        bboxs = [
            [
                (x - w/2) + offset[1],
                (y - h/2) + offset[0],
                (x + w/2) + offset[1],
                (y + h/2) + offset[0]
            ]
            for x, y, w, h in wh_boxes
        ]
        labels = results[0].boxes.cls.cpu().numpy().tolist()

        return bboxs, labels
