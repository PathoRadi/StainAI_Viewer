import numpy as np
import torch

class BoundingBoxFilter:
    """
    Static methods to filter boxes that lie too close to patch edges
    or to keep only those along certain “lines” in the patch.
    """
    @staticmethod
    def delete_edge(bboxs: torch.Tensor, labels: torch.Tensor, pl, pt, pr, pb):
        """
        Remove boxes that are too close to the patch edges.
        Args:
            bboxs: (N, 4) array of bounding boxes [x1, y1, x2, y2]
            labels: (N,) array of labels
            pl, pt, pr, pb: patch left, top, right, bottom coordinates
        Returns:
            Filtered boxes and labels as numpy arrays."""
        
        # Convert to numpy arrays for processing
        boxes = np.asarray(bboxs, dtype=np.float32)   
        labs  = np.asarray(labels)                   

        # Extract coordinates
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        # Create mask for boxes that are at least 8 pixels away from all edges
        mask = (
            (x1 - pl >= 8) &
            (y1 - pt >= 8) &
            (pr - x2 >= 8) &
            (pb - y2 >= 8)
        )

        return boxes[mask], labs[mask]

    @staticmethod
    def keep_cline(bboxs: torch.Tensor, labels: torch.Tensor, pl, pt, pr, pb):
        """
        Keep boxes that lie along the center cross lines of the patch.
        Args:
            bboxs: (N, 4) array of bounding boxes [x1, y1, x2, y2]
            labels: (N,) array of labels
            pl, pt, pr, pb: patch left, top, right, bottom coordinates
        Returns:
            Filtered boxes and labels as numpy arrays.
        """

        # Define the cross line boundaries
        width = 40
        # Horizontal line
        nl = pl + 320 - width
        # Vertical line
        nt = pt + 200
        # Horizontal line
        nr = pr - 320 + width
        # Vertical line
        nb = pb - 200

        # Convert to numpy arrays for processing
        boxes = np.asarray(bboxs, dtype=np.float32)
        labs  = np.asarray(labels)

        # Extract coordinates
        x1, y1, x2, y2 = boxes.T
        # Create mask for boxes that intersect with the cross lines
        mask = (x2 >= nl) & (x1 <= nr) & (y2 >= nt) & (y1 <= nb)

        return boxes[mask], labs[mask]

    @staticmethod
    def keep_rline(bboxs: torch.Tensor, labels: torch.Tensor, pl, pt, pr, pb):
        """
        Keep boxes that lie along the reverse cross lines of the patch.
        Args:
            bboxs: (N, 4) array of bounding boxes [x1, y1, x2, y2]
            labels: (N,) array of labels
            pl, pt, pr, pb: patch left, top, right, bottom coordinates
        Returns:
            Filtered boxes and labels as numpy arrays.
        """

        # Define the reverse cross line boundaries
        width = 40
        # Vertical line
        nl = pl + 200
        # Horizontal line
        nt = pt + 320 - width
        # Vertical line
        nr = pr - 200
        # Horizontal line
        nb = pb - 320 + width

        # Convert to numpy arrays for processing
        boxes = np.asarray(bboxs, dtype=np.float32)
        labs  = np.asarray(labels)

        # Extract coordinates
        x1, y1, x2, y2 = boxes.T
        # Create mask for boxes that intersect with the reverse cross lines
        mask = (x2 >= nl) & (x1 <= nr) & (y2 >= nt) & (y1 <= nb)

        return boxes[mask], labs[mask]

    @staticmethod
    def keep_center(bboxs: torch.Tensor, labels: torch.Tensor, pl, pt, pr, pb):
        """
        Keep boxes that lie within the central area of the patch.
        Args:
            bboxs: (N, 4) array of bounding boxes [x1, y1, x2, y2]
            labels: (N,) array of labels
            pl, pt, pr, pb: patch left, top, right, bottom coordinates
        Returns:
            Filtered boxes and labels as numpy arrays.
        """

        # Define the central area boundaries
        nl, nt = pl + 120, pt + 120
        nr, nb = pr - 120, pb - 120

        # Convert to numpy arrays for processing
        boxes = np.asarray(bboxs, dtype=np.float32)
        labs  = np.asarray(labels)

        # Extract coordinates
        x1, y1, x2, y2 = boxes.T
        # Calculate box centers
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        # Create mask for boxes whose centers lie within the central area
        mask = (nl <= cx) & (cx <= nr) & (nt <= cy) & (cy <= nb)

        return boxes[mask], labs[mask]