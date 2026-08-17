import numpy as np
import cv2
from typing import List, Tuple, Optional
import random

class EnhancedRICAP:
    """
    Enhanced RICAP data augmentation for multi-modal semantic segmentation.
 
    Parameters
    ----------
    H : int
        Number of horizontal slices (columns).
    V : int
        Number of vertical slices (rows).
    P : int
        Number of candidate proposals to generate; the one with the lowest
        matching cost is selected.
    alpha : float or None
        Weight balancing image-edge cost vs label-edge cost (Eq. 5 in paper).
        If None, it will be calibrated automatically via ``calibrate_alpha()``.
    random_crop_size : bool
        If True, each slice width/height is drawn randomly (paper default).
        If False, uniform slicing is used.
    """
 
    def __init__(
        self,
        H: int = 4,
        V: int = 4,
        P: int = 10,
        alpha: Optional[float] = None,
        random_crop_size: bool = True,
    ):
        self.H = H
        self.V = V
        self.P = P
        self.alpha = alpha
        self.random_crop_size = random_crop_size
 
    # ------------------------------------------------------------------
    # Alpha calibration (Eq. 6)
    # ------------------------------------------------------------------
    def calibrate_alpha(self, dataset, num_samples: int = 50):
        """
        Estimate alpha from a sample of the training data so that image-edge
        and label-edge matching costs have similar magnitudes (Eq. 6).
 
        Parameters
        ----------
        dataset : list or Dataset
            Each element should be a dict with at least:
              - "image": np.ndarray (H, W, 3) uint8/uint16  (RGB)
              - "mask" : np.ndarray (H, W) int/uint8         (label)
        num_samples : int
            Number of randomly-generated patched images used for calibration.
        """
        sum_image_edge = 0.0
        sum_label_edge = 0.0
        n = len(dataset)
 
        for _ in range(num_samples):
            # Build one random patched image just for edge statistics
            slices_w, slices_h = self._random_slices(
                dataset[0]["image"].shape[1],
                dataset[0]["image"].shape[0],
            )
 
            # Pick random images for each cell
            indices = [random.randint(0, n - 1) for _ in range(self.H * self.V)]
            images = [dataset[idx]["image"] for idx in indices]
            masks = [dataset[idx]["mask"] for idx in indices]
 
            # Compute edge costs along all internal boundaries
            img_cost, lbl_cost = self._compute_edge_costs_raw(
                images, masks, slices_w, slices_h
            )
            sum_image_edge += img_cost
            sum_label_edge += lbl_cost
 
        # Eq. 6:  alpha / (1 - alpha) = sum_image_edge / sum_label_edge
        if sum_label_edge < 1e-12:
            self.alpha = 0.5
        else:
            ratio = sum_image_edge / sum_label_edge
            self.alpha = ratio / (1.0 + ratio)
 
        print(f"[EnhancedRICAP] Calibrated alpha = {self.alpha:.4f}")
        return self.alpha
 
    # ------------------------------------------------------------------
    # Slice generation (Eq. 1-4)
    # ------------------------------------------------------------------
    def _random_slices(self, W: int, H_img: int) -> Tuple[List[int], List[int]]:
        """Return lists of slice widths and heights."""
        if self.random_crop_size:
            # Eq. 2-3: random ratios normalised to sum=1
            w_rand = np.random.uniform(0.1, 1.0, size=self.H)
            h_rand = np.random.uniform(0.1, 1.0, size=self.V)
            w_ratios = w_rand / w_rand.sum()
            h_ratios = h_rand / h_rand.sum()
        else: # uniform slicing
            w_ratios = np.ones(self.H) / self.H
            h_ratios = np.ones(self.V) / self.V
 
        # Eq. 1: convert ratios to pixel sizes (ensure they sum exactly to W/H) # Fix rounding residuals
        slices_w = np.round(w_ratios * W).astype(int)
        slices_h = np.round(h_ratios * H_img).astype(int)
        
        slices_w[-1] = W - slices_w[:-1].sum()
        slices_h[-1] = H_img - slices_h[:-1].sum()
 
        return slices_w.tolist(), slices_h.tolist()
 
    # ------------------------------------------------------------------
    # Edge cost computation (Eq. 5, 7)
    # ------------------------------------------------------------------
    @staticmethod
    def _edge_diff_image(img_left: np.ndarray, img_right: np.ndarray) -> float:
        """L2 difference of the rightmost column of img_left vs leftmost column of img_right."""
        # Both are float32 [H_slice, C] or [H_slice]
        left_col = img_left[:, -1].astype(np.float64)
        right_col = img_right[:, 0].astype(np.float64)
        return np.sum((left_col - right_col) ** 2)
 
    @staticmethod
    def _edge_diff_mask(mask_left: np.ndarray, mask_right: np.ndarray) -> float:
        """L2 difference of the rightmost column of mask_left vs leftmost column of mask_right."""
        left_col = mask_left[:, -1].astype(np.float64)
        right_col = mask_right[:, 0].astype(np.float64)
        return np.sum((left_col - right_col) ** 2)
 
    def _compute_edge_costs_raw(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        slices_w: List[int],
        slices_h: List[int],
    ) -> Tuple[float, float]:
        """
        Compute raw (un-weighted) total image-edge cost and label-edge cost
        for a given patching arrangement.
 
        images/masks are in row-major order: index = i * V + j
            where i is the column index (0..H-1) and j is the row index (0..V-1).
        """
        W = sum(slices_w)
        H_img = sum(slices_h)
 
        total_img_cost = 0.0
        total_lbl_cost = 0.0
 
        # Crop each cell
        def _crop_cell(full_img, ci, cj):
            """Crop the (ci, cj) region from full_img."""
            x0 = sum(slices_w[:ci])
            y0 = sum(slices_h[:cj])
            w = slices_w[ci]
            h = slices_h[cj]
            if full_img.ndim == 3:
                return full_img[y0:y0+h, x0:x0+w, :]
            else:
                return full_img[y0:y0+h, x0:x0+w]
 
        # Horizontal edges (between column i and i+1)
        for ci in range(self.H - 1):
            for cj in range(self.V):
                idx_left = ci * self.V + cj
                idx_right = (ci + 1) * self.V + cj
                img_l = _crop_cell(images[idx_left], ci, cj)
                img_r = _crop_cell(images[idx_right], ci + 1, cj)
                msk_l = _crop_cell(masks[idx_left], ci, cj)
                msk_r = _crop_cell(masks[idx_right], ci + 1, cj)
 
                if img_l.shape[0] > 0 and img_r.shape[0] > 0:
                    total_img_cost += self._edge_diff_image(img_l, img_r)
                    total_lbl_cost += self._edge_diff_mask(msk_l, msk_r)
 
        # Vertical edges (between row j and j+1)
        for ci in range(self.H):
            for cj in range(self.V - 1):
                idx_top = ci * self.V + cj
                idx_bot = ci * self.V + (cj + 1)
                img_t = _crop_cell(images[idx_top], ci, cj)
                img_b = _crop_cell(images[idx_bot], ci, cj + 1)
                msk_t = _crop_cell(masks[idx_top], ci, cj)
                msk_b = _crop_cell(masks[idx_bot], ci, cj + 1)
 
                if img_t.shape[0] > 0 and img_b.shape[0] > 0:
                    # Bottom row of top cell vs top row of bottom cell
                    total_img_cost += np.sum(
                        (img_t[-1, :].astype(np.float64) - img_b[0, :].astype(np.float64)) ** 2
                    )
                    total_lbl_cost += np.sum(
                        (msk_t[-1, :].astype(np.float64) - msk_b[0, :].astype(np.float64)) ** 2
                    )
 
        return total_img_cost, total_lbl_cost
 
    def _matching_cost(self, cell_rgb, cell_mask) -> float:
        img_cost = 0.0
        lbl_cost = 0.0
        for ci in range(self.H - 1):
            for cj in range(self.V):
                img_cost += self._edge_diff_image(cell_rgb[(ci, cj)], cell_rgb[(ci + 1, cj)])
                lbl_cost += self._edge_diff_image(cell_mask[(ci, cj)], cell_mask[(ci + 1, cj)])
        for ci in range(self.H):
            for cj in range(self.V - 1):
                top_r = cell_rgb[(ci, cj)]
                bot_r = cell_rgb[(ci, cj + 1)]
                top_m = cell_mask[(ci, cj)]
                bot_m = cell_mask[(ci, cj + 1)]
                img_cost += np.sum((top_r[-1, :].astype(np.float64) - bot_r[0, :].astype(np.float64)) ** 2)
                lbl_cost += np.sum((top_m[-1, :].astype(np.float64) - bot_m[0, :].astype(np.float64)) ** 2)
        alpha = self.alpha if self.alpha is not None else 0.5
        return (1 - alpha) * img_cost + alpha * lbl_cost
    
    # ------------------------------------------------------------------
    # _apply_one: Apply a given patching arrangement to a list of images/masks
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_one(img_all, indices, slices_w, slices_h):
        patched = np.empty_like(img_all[0])
        y0 = 0
        cell_idx = 0
        for cj, h in enumerate(slices_h):
            x0 = 0
            for ci, w in enumerate(slices_w):
                src = indices[cell_idx]
                cell_idx += 1
                if patched.ndim == 3:
                    patched[y0:y0+h, x0:x0+w, :] = img_all[src][y0:y0+h, x0:x0+w, :]
                else:
                    patched[y0:y0+h, x0:x0+w] = img_all[src][y0:y0+h, x0:x0+w]
                x0 += w
            y0 += h
        return patched
    
    # ------------------------------------------------------------------
    # generate: Generate a patched image/mask from multiple modalities
    # ------------------------------------------------------------------
    def generate(
        self,
        rgb_all: List[np.ndarray],
        mask_all: List[np.ndarray],
        *extra_modals: List[np.ndarray],
    ) -> tuple:
        """
        Optimal single-pass completion: Select a proposal using RGB and labels, then apply it across all modalities.

        Parameters
        ----------
        rgb_all : list of np.ndarray (H, W, 3)
        mask_all : list of np.ndarray (H, W)
        *extra_modals : list of np.ndarray (H, W) for additional modalities, e.g., nir_all, re_all, ndvi_all
 
        Returns
        -------
        tuple: (patched_rgb, patched_mask, patched_extra1, patched_extra2, ...)
               Order matches the input order.
        """
        n = len(rgb_all)
        H_img, W = rgb_all[0].shape[:2]
 
        best_cost = float("inf")
        best_slices_w = None
        best_slices_h = None
        best_indices = None
 
        for _ in range(self.P):
            slices_w, slices_h = self._random_slices(W, H_img)
            indices = [random.randint(0, n - 1) for _ in range(self.H * self.V)]
 
            cell_rgb = {}
            cell_mask = {}
            y0 = 0
            cell_idx = 0
            for cj, h in enumerate(slices_h):
                x0 = 0
                for ci, w in enumerate(slices_w):
                    src = indices[cell_idx]
                    cell_rgb[(ci, cj)] = rgb_all[src][y0:y0+h, x0:x0+w]
                    cell_mask[(ci, cj)] = mask_all[src][y0:y0+h, x0:x0+w]
                    cell_idx += 1
                    x0 += w
                y0 += h
 
            cost = self._matching_cost(cell_rgb, cell_mask)
            if cost < best_cost:
                best_cost = cost
                best_slices_w = slices_w
                best_slices_h = slices_h
                best_indices = indices
 
        # Apply the best proposal to all modalities
        out_rgb  = self._apply_one(rgb_all,  best_indices, best_slices_w, best_slices_h)
        out_mask = self._apply_one(mask_all, best_indices, best_slices_w, best_slices_h)
        out_extras = tuple(
            self._apply_one(modal, best_indices, best_slices_w, best_slices_h)
            for modal in extra_modals
        )
 
        return (out_rgb, out_mask) + out_extras

# if __name__ == "__main__":
#     print(f"Calibrating aplha on {len(train_df)} training samples...")

#     dataset= []
#     for i in range(len(train_df)):
#         img_name = train_df.iloc[i]["image"]
#         rgb_file = rf"{rgb_path}\{img_name}.png"
#         mask_file = rf"{label_path}\{img_name}.png"

#         img = cv2.imread(rgb_file,cv2.IMREAD_UNCHANGED)
#         if img is None:
#             print(f"Warning: Could not read image file {rgb_file}")
#             continue
#         if img.dtype == np.uint16:
#             img = (img/256).astype(np.uint8)
#         mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             print(f"Warning: Could not read mask file {mask_file}")
#             continue

#         dataset.append({"image":img,"mask":mask})
#     print(f"Loaded {len(dataset)} valid training samples for calibration.")
#     print(f"Image shape: {dataset[0]['image'].shape}, Mask shape: {dataset[0]['mask'].shape}")

#     ricap = EnhancedRICAP(H=4, V=4, P=10, random_crop_size=True)
#     alpha = ricap.calibrate_alpha(dataset, num_samples=50)

#     print(f"Grid size: {ricap.H}x{ricap.V}")
#     print(f"Calibrated alpha: {alpha:.4f}")