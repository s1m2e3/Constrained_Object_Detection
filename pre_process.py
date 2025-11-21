import cv2
import numpy as np
import torch
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte


def apply_gaussian_filter(image_batch, num_gaussians=1):
    """
    Applies two different Gaussian filters to a batch of images.
    """
    processed_batch = []
    kernels = [((5, 5), 0.6), ((9, 9), 1.3), ((11, 11), 1.6)] # (ksize, sigma)
    kernels = kernels[:num_gaussians]
    for img in image_batch:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) > 2 else img

        for ksize, sigma in kernels:
            img = cv2.GaussianBlur(gray_img, ksize, sigma).astype(float)
            min_val = np.min(img)
            max_val = np.max(img)
            img = (img - min_val) / (max_val - min_val)

            processed_batch.append(img)
    return processed_batch

def apply_laplacian_filter(image_batch, num_gaussians=1, num_laplacians=1):
    """
    Applies a Gaussian filter followed by a Laplacian filter to a batch of images.
    """
    k_size = 3
    processed_batch = []
    gaussian_filtered = apply_gaussian_filter(image_batch, num_gaussians)
    for img in gaussian_filtered:
        laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=k_size).astype(float)
        laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())
        # Convert back to 8-bit image
        processed_batch.append(cv2.convertScaleAbs(laplacian))
    return processed_batch

def apply_gabor_filter(image_batch, num_gabor=6):
    """
    Applies a bank of Gabor filters to a batch of images.
    """
    
    processed_batch = []
    # A set of Gabor filter parameters
    params = [
        {'ksize': (11, 11), 'sigma': 5, 'theta': 0, 'lambd': 10.0, 'gamma': 0.5},
        {'ksize': (11, 11), 'sigma': 5, 'theta': np.pi / 4, 'lambd': 10.0, 'gamma': 0.5},
        {'ksize': (11, 11), 'sigma': 5, 'theta': np.pi / 2, 'lambd': 10.0, 'gamma': 0.5},
        {'ksize': (11, 11), 'sigma': 5, 'theta': 3 * np.pi / 4, 'lambd': 10.0, 'gamma': 0.5},
        {'ksize': (21, 21), 'sigma': 9, 'theta': np.pi / 2, 'lambd': 15.0, 'gamma': 0.7},
        {'ksize': (21, 21), 'sigma': 9, 'theta': 3 * np.pi / 4, 'lambd': 15.0, 'gamma': 0.7}
    ]
    params = params[:num_gabor]

    for img in image_batch:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) > 2 else img
        accumulated_gabor = np.zeros(gray_img.shape)
        for p in params:
            gabor_kernel = cv2.getGaborKernel(ksize=p['ksize'], sigma=p['sigma'], theta=p['theta'],
                                              lambd=p['lambd'], gamma=p['gamma'])
            filtered_img = cv2.filter2D(gray_img, cv2.CV_8U, gabor_kernel)
            accumulated_gabor += np.abs(filtered_img)**2
        accumulated_gabor = np.sqrt(accumulated_gabor/len(params))
        accumulated_gabor = ((accumulated_gabor - accumulated_gabor.min()) / (accumulated_gabor.max() - accumulated_gabor.min()))**(3)

        processed_batch.append(accumulated_gabor)
    return processed_batch


def apply_sobel_filter(image_batch):
    """
    Applies the Sobel filter to a batch of images.
    """
    processed_batch = []
    for img in image_batch:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) > 2 else img
        
        # Use float for gradients
        grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-255 and convert to uint8 for viewing
        grad_mag_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
        grad_img = grad_mag_norm.astype(float)

        grad_img = ((grad_img - grad_img.min()) / (grad_img.max() - grad_img.min()))

        processed_batch.append(grad_img)
    return processed_batch


def apply_ndi(image_batch):
    """
    Applies the NDI filter to a batch of images.
    """
    processed_batch = []
    for img in image_batch:
        img = img.astype(np.float32) / 255.0
        red = img[:, :, 0]
        green = img[:, :, 1]
        ndi =  (red - green)/(red+green+1e-5)
        min_ndi = np.min(ndi)
        max_ndi = np.max(ndi)
        ndi = ((ndi - min_ndi) / (max_ndi - min_ndi))**2
        processed_batch.append(ndi)
    return processed_batch

def chromaticity(image_batch):
    """
    Applies the Chromaticity filter to a batch of images.
    """
    processed_batch = []
    for img in image_batch:        
        img = img.astype(np.float32) / 255.0
        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]
        chromaticity = (red)/(red+green+blue+1e-5)
        min_chromaticity = np.min(chromaticity)
        max_chromaticity = np.max(chromaticity)
        chromaticity = ((chromaticity - min_chromaticity) / (max_chromaticity - min_chromaticity))**3
        
        processed_batch.append(chromaticity)
    return processed_batch



def apply_grayness(image_batch):
    """
    Applies the Grayness filter to a batch of images.
    """
    processed_batch = []
    for img in image_batch:
        img = img.astype(np.float32) / 255.0
        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]
        grayness = 1-(np.sqrt((red-green)**2+(green-blue)**2+(blue-red)**2))/np.sqrt(3)
        min_grayness = np.min(grayness)
        max_grayness = np.max(grayness)
        grayness = ((grayness - min_grayness) / (max_grayness - min_grayness))**4

        processed_batch.append(grayness)
    return processed_batch


def create_polynomial_features(image_batch):
    processed_batch = []
    for img in image_batch:
        img = img.astype(np.float32) / 255.0
        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]
        rr = red**2
        rr_min = np.min(rr)
        rr_max = np.max(rr)
        rr = (rr-rr_min)/(rr_max-rr_min)
        gg = green**2
        gg_min = np.min(gg)
        gg_max = np.max(gg)
        gg = (gg-gg_min)/(gg_max-gg_min)
        bb = blue**2
        bb_min = np.min(bb)
        bb_max = np.max(bb)
        bb = (bb-bb_min)/(bb_max-bb_min)
        energy = red**2 + green**2 + blue**2
        energy_min = np.min(energy)
        energy_max = np.max(energy)
        energy = (energy-energy_min)/(energy_max-energy_min)
        rg = red*green
        rg_min = np.min(rg)
        rg_max = np.max(rg)
        rg = (rg-rg_min)/(rg_max-rg_min)
        rb = red*blue
        rb_min = np.min(rb)
        rb_max = np.max(rb)
        rb = (rb-rb_min)/(rb_max-rb_min)
        gb = green*blue
        gb_min = np.min(gb)
        gb_max = np.max(gb)
        gb = (gb-gb_min)/(gb_max-gb_min)
        rgb = red*green*blue
        rgb_min = np.min(rgb)
        rgb_max = np.max(rgb)
        rgb = (rgb-rgb_min)/(rgb_max-rgb_min)

        br_ratio = (blue-red)/(blue + red + 1e-5)
        br_ratio_min = np.min(br_ratio)
        br_ratio_max = np.max(br_ratio)
        br_ratio = ((br_ratio-br_ratio_min)/(br_ratio_max-br_ratio_min))**2

        bg_ratio = (blue-green)/(blue + green + 1e-5)
        bg_ratio_min = np.min(bg_ratio)
        bg_ratio_max = np.max(bg_ratio)
        bg_ratio = ((bg_ratio-bg_ratio_min)/(bg_ratio_max-bg_ratio_min))**2

        rg_ratio = (red-green)/(red + green + 1e-5)
        rg_ratio_min = np.min(rg_ratio)
        rg_ratio_max = np.max(rg_ratio) 
        rg_ratio = ((rg_ratio-rg_ratio_min)/(rg_ratio_max-rg_ratio_min))**2

        rb_ratio = (red-blue)/(red + blue + 1e-5)
        rb_ratio_min = np.min(rb_ratio)
        rb_ratio_max = np.max(rb_ratio)
        rb_ratio = ((rb_ratio-rb_ratio_min)/(rb_ratio_max-rb_ratio_min))**2

        gb_ratio = (green-blue)/(green + blue + 1e-5)
        gb_ratio_min = np.min(gb_ratio)
        gb_ratio_max = np.max(gb_ratio)
        gb_ratio = ((gb_ratio-gb_ratio_min)/(gb_ratio_max-gb_ratio_min))**2

        gr_ratio = (green-red)/(green + red + 1e-5)
        gr_ratio_min = np.min(gr_ratio)
        gr_ratio_max = np.max(gr_ratio)
        gr_ratio = ((gr_ratio-gr_ratio_min)/(gr_ratio_max-gr_ratio_min))**2

        processed_batch.append(np.stack([energy,br_ratio, bg_ratio, rg_ratio, rb_ratio, gb_ratio, gr_ratio, rgb], axis=-1))
    return processed_batch

def apply_entropy(image_batch):
    """
    Applies the Entropy filter to a batch of images.
    """
    processed_batch = []
    for image in image_batch:
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray_8bit = img_as_ubyte(gray)  # convert to uint8 if needed
        entropy_map = entropy(gray_8bit, disk(3))
        processed_batch.append(entropy_map)
    return processed_batch

def apply_green_normalized_index(image_batch):
    """
    Applies the Green Normalized Index filter to a batch of images.
    """
    processed_batch = []
    for img in image_batch:        
        img = img.astype(np.float32) / 255.0
        green = img[:, :, 1]
        blue = img[:, :, 2]
        red = img[:, :, 0]
        normalized_green = (green)/(green+blue+red+1e-5)
        min_normalized_green = np.min(normalized_green)
        max_normalized_green = np.max(normalized_green)
        normalized_green = ((normalized_green - min_normalized_green) / (max_normalized_green - min_normalized_green))**2
        processed_batch.append(normalized_green)
    return processed_batch

def apply_blue_normalized_index(image_batch):
    """
    Applies the Blue Normalized Index filter to a batch of images.
    """
    processed_batch = []
    for img in image_batch:
        img = img.astype(np.float32) / 255.0
        green = img[:, :, 1]
        blue = img[:, :, 2]
        red = img[:, :, 0]
        normalized_blue = (blue)/(green+blue+red+1e-5)
        min_normalized_blue = np.min(normalized_blue)
        max_normalized_blue = np.max(normalized_blue)
        normalized_blue = ((normalized_blue - min_normalized_blue) / (max_normalized_blue - min_normalized_blue))**2
        processed_batch.append(normalized_blue)
    return processed_batch



def apply_filters(image_batch,num_gaussians=1,num_laplacians=1,num_gabor=6):
    """
    Applies a series of filters to the image batch and stacks the results.
    Note: The number of filters applied is determined by the implementations
    of the filter functions, not the parameters of this function.
    """
    if image_batch.shape[0]==0:
        return np.array([])
    
    # All outputs should be flat: a list of (H, W) arrays
    def flatten(batch):
        """Takes a batch of (H,W,C) or (H,W) and returns a list of (H,W) slices"""
        if isinstance(batch, list):
            result = []
            for item in batch:
                if item.ndim == 2:
                    result.append(item)
                elif item.ndim == 3:
                    # split along channel
                    result.extend([item[..., c] for c in range(item.shape[-1])])
                else:
                    raise ValueError("Unexpected shape in batch:", item.shape)
            return result
        else:
            raise ValueError("Input to flatten must be a list")

    all_processed = np.stack((
        np.stack(flatten(sobel_processed := apply_sobel_filter(image_batch)) ,axis=0),
        np.stack(flatten(laplacian_processed := apply_laplacian_filter(image_batch, num_laplacians)) ,axis=0),
        np.stack(flatten(grayness_processed := apply_grayness(image_batch)) ,axis=0),
        np.stack(flatten(ndi_processed := apply_ndi(image_batch)) ,axis=0),
        np.stack(flatten(entropy_ndi := apply_gaussian_filter(apply_entropy(image_batch))) ,axis=0),
        np.stack(flatten(gabor_processed := apply_gaussian_filter(apply_gabor_filter(image_batch, num_gabor))) ,axis=0),
        np.stack(flatten(sobel_grayness := apply_gaussian_filter(apply_sobel_filter(apply_grayness(image_batch)))) ,axis=0),
        np.stack(flatten(entropy_gabor := apply_gaussian_filter(apply_entropy(apply_gabor_filter(image_batch, num_gabor)))) ,axis=0),
        np.stack(flatten(blue_normalized_index := apply_blue_normalized_index(image_batch)) ,axis=0),
        np.stack(flatten(green_normalized_index := apply_green_normalized_index(image_batch)) ,axis=0),
        
    ),axis=3)
    polynomial_batch = np.stack(create_polynomial_features(image_batch),axis=0)
    all_processed = np.concatenate((all_processed,polynomial_batch),axis=3)
    return all_processed

def binarize_objectness(image_batch,label_batch,num_classes=12):
    H,W,_ = image_batch[0].shape
    labeled_images = []
    for index in range(len(label_batch)):
        binary_image = np.zeros((H,W,num_classes))
        for objects in label_batch[index]:
            x1 = int(objects[1][0])
            y1 = int(objects[1][1])
            x2 = int(objects[1][0])+int(objects[1][2]) 
            y2 = int(objects[1][1])+int(objects[1][3]) 
            binary_image[y1:y2,x1:x2,objects[0]] = 1
        labeled_images.append(binary_image)
    
    scored_labeled_images = score_patches(np.array(labeled_images))
    
    return scored_labeled_images

def score_patches(labeled_images,channel_last=True,stride=1,pad=1,sliding_window_size=3):
    tensor_images = torch.from_numpy(labeled_images)
    
    x = tensor_images.permute(0, 3, 1, 2) if channel_last else tensor_images  # (B, C, H, W)
    B, C, H, W = x.shape
    

    cols = torch.nn.functional.unfold(x, kernel_size=3, stride=stride, padding=pad)  # (B, C*9, L)
    cols = cols.view(B, C, int(sliding_window_size**2), -1)                                  # (B, C, 9, L)
    cov_true  = cols.mean(dim=2)                                        # (B, C, L)
    
     # Compute spatial grid size
    H_out = (H + 2*pad - sliding_window_size) // stride + 1
    W_out = (W + 2*pad - sliding_window_size) // stride + 1

    # Reshape back to grids
    cov_true_grid  = cov_true.view(B, C, H_out, W_out)
    # cov_false_grid = cov_false.view(B, C, H_out, W_out)
    
    return cov_true_grid.numpy()


