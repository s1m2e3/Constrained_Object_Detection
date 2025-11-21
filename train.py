from import_dataset import batched_import
from pre_process import apply_filters, binarize_objectness
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import itertools
from model import SimpleModel
import os
import random

def visualize_image(image_original, image_processed, title="Image", cmap=None):
    """
    Displays an image using matplotlib.

    Args:
        image (np.array): The image to display.
        title (str): The title for the image plot.
        cmap (str, optional): The colormap to use for grayscale images.
                              Defaults to None.
    """
    image_original_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    
    fig, ax = plt.subplots(2, 4, figsize=(10, 20))
    for i in range(2):
        for j in range(4):
            if i==0 and j == 0:
                ax[i,j].imshow(image_original, cmap=cmap)
            if i == 0 and j != 0:
                ax[i,j].imshow(image_processed[:,:,j+8], cmap='gray')
            if i == 1 and j ==0:
                ax[i,j].imshow(image_original_gray, cmap='gray')
            if i ==1 and j !=0:
                ax[i,j].imshow(image_original_gray-image_processed[:,:,j+8], cmap='gray')
            ax[i,j].set_title(f"{title} {i} {j}")
            ax[i,j].axis('off')
    plt.show()

def create_patches(image_tensor, patch_size=32, stride=None):
    """
    Creates patches from an image tensor using PyTorch.

    Args:
        image_tensor (torch.Tensor): The input image tensor of shape (C, H, W).
        patch_size (int): The size of the square patches (K).
        stride (int, optional): The step size between patches. 
                                Defaults to patch_size for non-overlapping patches.

    Returns:
        torch.Tensor: A tensor of patches with shape (num_patches, C, patch_size, patch_size).
    """
    if stride is None:
        stride = patch_size
    
    # Unfold the image tensor to create patches
    # The tensor is expected to be in (C, H, W) format
    patches = image_tensor.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    # The shape after unfolding is (C, num_patches_h, num_patches_w, patch_size, patch_size)
    
    # Reshape to (num_patches, C, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, image_tensor.shape[0], patch_size, patch_size)
    
    return patches


def reconstruct_from_patches(patches, original_h, original_w, stride=None, patch_size=32):
    """
    Reconstructs an image from a tensor of patch summary values using PyTorch.
    It handles overlapping patches by averaging.

    Args:
        patches (torch.Tensor): A tensor of patch values with shape (num_patches,).
        original_h (int): The height of the original image.
        original_w (int): The width of the original image.
        stride (int, optional): The step size used when creating the patches.
                                Defaults to patch_size for non-overlapping patches.
        patch_size (int): The size of the square patches.

    Returns:
        torch.Tensor: The reconstructed image tensor of shape (C, original_h, original_w).
    """
    num_patches = patches.shape[0]
    C = 1  # We are reconstructing a single-channel image.

    if stride is None:
        stride = patch_size

    # Create a Fold operation to stitch the patches back together
    fold = torch.nn.Fold(output_size=(original_h, original_w),
                         kernel_size=(patch_size, patch_size),
                         stride=(stride, stride))

    # Expand each patch value to a full patch of size (C, patch_size, patch_size)
    # The shape for fold needs to be (B, C * K * K, L) where L is num_patches
    patches_for_fold = patches.view(1, 1, num_patches).repeat(1, C * patch_size * patch_size, 1)

    # The fold operation returns a tensor of shape (B, C, H, W), so we squeeze the batch dimension
    return fold(patches_for_fold).squeeze(0)


def grad_global_norm(model, p=2):
    norms = []
    for p_ in model.parameters():
        if p_.grad is not None:
            norms.append(p_.grad.detach().data.norm(p))
    if not norms: 
        return 0.0
    # L2 global norm: sqrt(sum(norm_i^2)); for p!=2 you’d aggregate accordingly.
    if p == 2:
        return torch.sqrt(torch.sum(torch.stack([n**2 for n in norms])))
    elif p == float('inf'):
        return torch.max(torch.stack(norms))
    else:
        # generalized p: (sum ||g_i||_p^p)^(1/p)
        return torch.pow(torch.sum(torch.stack([n**p for n in norms])), 1.0/p)




def train_loop(model,optimizer,x_image,y,device,patch_size=16,stride=16,weights_path='weights/SimpleModel.pt'):
    
    optimizer.zero_grad()
    x_image_np = x_image
    x_image = torch.from_numpy(x_image).float().permute(0, 3, 1, 2).to(device)
    y = torch.from_numpy(y).float().to(device)
    

    x = model.patch(x_image,patch_size,stride)
    y = model.patch(y,patch_size,stride).amax(dim=-1).amax(dim=-1).amax(dim=-1).unsqueeze(-1)
    
    tp = torch.where(y>0)
    if tp[0].shape[0]>0:
        tn = torch.where(y==0)
        pred = model(x)

        pred_normalized = model.normalize_prediction(pred)
        pred = model.eval_constraints(pred)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        logits_loss = loss_fn(pred,y)+1e-3*(pred.sum(-1).amax()**2+pred.sum(-1).amin()**2)
        loss = logits_loss
        loss.backward()
        total_gnorm = grad_global_norm(model, p=2)
        print('gnorm is: ',np.round(total_gnorm.item(),4))
        print('loss is: ',np.round(logits_loss.item(),4))
        optimizer.step()
        torch.save(model.state_dict(), weights_path)
        print(f"Saved model weights to {weights_path}")
        # pred = pred.sum(dim=-1)
        # pred = (pred - pred.mean())/pred.std()
        # pred_bin = torch.where(pred>0,1,0)
        # y = y.amax(dim=-1)
        # pred = model.unpatch(pred,patch_size,stride)
        # y = model.unpatch(y,patch_size,stride)
        # x = x.amax(dim=(-1,-2))
        # x_grayness = model.unpatch((x[:,:,:,2]*pred_bin),patch_size,stride)
        # x_grayness = (x_grayness - x_grayness.amin(dim=(-1,-2),keepdim=True)) / (x_grayness.amax(dim=(-1,-2),keepdim=True) - x_grayness.amin(dim=(-1,-2),keepdim=True))
        # x_ndi_entropy = model.unpatch((x[:,:,:,4]*pred_bin),patch_size,stride)
        # x_ndi_entropy = (x_ndi_entropy- x_ndi_entropy.amin(dim=(-1,-2),keepdim=True)) / (x_ndi_entropy.amax(dim=(-1,-2),keepdim=True)  - x_ndi_entropy.amin(dim=(-1,-2),keepdim=True))
        # x_blue_index = model.unpatch((x[:,:,:,8]*pred_bin),patch_size,stride)
        # x_blue_index = (x_blue_index- x_blue_index.amin(dim=(-1,-2),keepdim=True)) / (x_blue_index.amax(dim=(-1,-2),keepdim=True)  - x_blue_index.amin(dim=(-1,-2),keepdim=True))
        # x_green = model.unpatch(((x[:,:,:,9]+x[:,:,:,-6]+x[:,:,:,3])*pred_bin),patch_size,stride)
        # x_green = (x_green- x_green.amin(dim=(-1,-2),keepdim=True)) / (x_green.amax(dim=(-1,-2),keepdim=True)  - x_green.amin(dim=(-1,-2),keepdim=True))
        # x_edges = model.unpatch(((x[:,:,:,0]+x[:,:,:,1])*pred_bin),patch_size,stride)
        # x_edges = (x_edges- x_edges.amin(dim=(-1,-2),keepdim=True)) / (x_edges.amax(dim=(-1,-2),keepdim=True)  - x_edges.amin(dim=(-1,-2),keepdim=True))
        # x_gabor = model.unpatch(((x[:,:,:,5])*pred_bin),patch_size,stride)
        # x_gabor = (x_gabor- x_gabor.amin(dim=(-1,-2),keepdim=True)) / (x_gabor.amax(dim=(-1,-2),keepdim=True)  - x_gabor.amin(dim=(-1,-2),keepdim=True))

        # x_grayness = x_grayness.detach().cpu().numpy()
        # x_ndi_entropy = x_ndi_entropy.detach().cpu().numpy()
        # x_blue_index = x_blue_index.detach().cpu().numpy()
        # x_green = x_green.detach().cpu().numpy()
        # x_edges = x_edges.detach().cpu().numpy()
        # x_gabor = x_gabor.detach().cpu().numpy()

        # grayness_low = np.where(x_grayness<np.percentile(x_grayness,40),1.0,0.0)
        # grayness_high = np.where(x_grayness>np.percentile(x_grayness,80),1.0,0.0)
        # grayness_medium = np.where((x_grayness>np.percentile(x_grayness,40)) & (x_grayness<np.percentile(x_grayness,80)),1.0,0.0)
        # ndi_entropy_low = np.where(x_ndi_entropy<np.percentile(x_ndi_entropy,40),1.0,0.0)
        # ndi_entropy_high = np.where(x_ndi_entropy>np.percentile(x_ndi_entropy,80),1.0,0.0)
        # ndi_entropy_medium = np.where((x_ndi_entropy>np.percentile(x_ndi_entropy,40)) & (x_ndi_entropy<np.percentile(x_ndi_entropy,80)),1.0,0.0)

        # blue_index_low = np.where(x_blue_index<np.percentile(x_blue_index,40),1.0,0.0)
        # blue_index_high = np.where(x_blue_index>np.percentile(x_blue_index,80),1.0,0.0)
        # blue_index_medium = np.where((x_blue_index>np.percentile(x_blue_index,40)) & (x_blue_index<np.percentile(x_blue_index,80)),1.0,0.0)

        # green_low = np.where(x_green<np.percentile(x_green,40),1.0,0.0)
        # green_high = np.where(x_green>np.percentile(x_green,90),1.0,0.0)
        # green_medium = np.where((x_green>np.percentile(x_green,40)) & (x_green<np.percentile(x_green,90)),1.0,0.0)

        # edges_low = np.where(x_edges<np.percentile(x_edges,40),1.0,0.0)
        # edges_high = np.where(x_edges>np.percentile(x_edges,80),1.0,0.0)
        # edges_medium = np.where((x_edges>np.percentile(x_edges,40)) & (x_edges<np.percentile(x_edges,80)),1.0,0.0)

        # gabor_low = np.where(x_gabor<np.percentile(x_gabor,40),1.0,0.0)
        # gabor_high = np.where(x_gabor>np.percentile(x_gabor,80),1.0,0.0)
        # gabor_medium = np.where((x_gabor>np.percentile(x_gabor,40)) & (x_gabor<np.percentile(x_gabor,80)),1.0,0.0)

        # sky_ground = (blue_index_medium+blue_index_high+green_low+green_medium)*(ndi_entropy_low+ndi_entropy_medium)
        # trees = (green_high)*(ndi_entropy_high)
        # edges_and_texture = (edges_medium+edges_high)*(gabor_high)
        # edges_and_not_texture = (edges_high)*(gabor_medium)
        
        # added_filters = (sky_ground+trees+edges_and_texture)
        # pred_bin = model.unpatch(pred_bin,patch_size,stride)
        # tensor_added_filters = torch.from_numpy(added_filters).float().to(device)
        # substracted_pred = pred_bin*(1-tensor_added_filters)
        # substracted_pred = (substracted_pred- substracted_pred.amin(dim=(-1,-2),keepdim=True)) / (substracted_pred.amax(dim=(-1,-2),keepdim=True)  - substracted_pred.amin(dim=(-1,-2),keepdim=True))
        # normalized_substracted_pred = substracted_pred.detach().cpu().numpy()
        # normalized_substracted_pred = np.where(normalized_substracted_pred>0.8,1.0,0.0)

        # fig, ax = plt.subplots(6, 4, figsize=(10, 20))
        # ax[0,0].imshow((x_image_np[0,:,:,-3:]*255).astype(np.uint8), cmap=None)
        # ax[0,0].axis('off')
        # ax[0,1].imshow(pred[0].detach().cpu().numpy(), cmap='viridis',alpha=1.0)
        # ax[0,1].imshow((x_image_np[0,:,:,-3:]*255).astype(np.uint8), cmap=None,alpha=0.7)
        # ax[0,1].axis('off')
        # ax[0,2].imshow(y[0].detach().cpu().numpy(), cmap='viridis',alpha=1.0)
        # ax[0,2].imshow((x_image_np[0,:,:,-3:]*255).astype(np.uint8), cmap=None,alpha=0.7)
        # ax[0,2].axis('off')

        # # TP
        # ax[0,3].imshow((pred_bin[0].detach().cpu().numpy()*y[0].detach().cpu().numpy()*255).astype(np.uint8), cmap=None)
        # ax[0,3].axis('off')


        # ax[1,3-3].imshow((sky_ground[0]*255).astype(np.uint8), cmap=None)
        # ax[1,3-3].axis('off')
        # ax[1,4-3].imshow((trees[0]*255).astype(np.uint8), cmap=None)
        # ax[1,4-3].axis('off')
        # ax[1,5-3].imshow((edges_and_texture[0]*255).astype(np.uint8), cmap=None)
        # ax[1,5-3].axis('off')
        # ax[1,6-3].imshow((edges_and_not_texture[0]*255).astype(np.uint8), cmap=None)
        # ax[1,6-3].axis('off')

        # ax[2,3-3].imshow(((y[0].detach().cpu().numpy())*sky_ground[0]*255).astype(np.uint8), cmap=None)
        # ax[2,3-3].axis('off')
        # ax[2,4-3].imshow(((y[0].detach().cpu().numpy())*trees[0]*255).astype(np.uint8), cmap=None)
        # ax[2,4-3].axis('off')
        # ax[2,5-3].imshow(((y[0].detach().cpu().numpy())*edges_and_texture[0]*255).astype(np.uint8), cmap=None)
        # ax[2,5-3].axis('off')
        # ax[2,6-3].imshow(((y[0].detach().cpu().numpy())*edges_and_not_texture[0]*255).astype(np.uint8), cmap=None)
        # ax[2,6-3].axis('off')

        # ax[3,3-3].imshow(((pred[0].detach().cpu().numpy())*sky_ground[0]*255).astype(np.uint8), cmap=None)
        # ax[3,3-3].axis('off')
        # ax[3,4-3].imshow(((pred[0].detach().cpu().numpy())*trees[0]*255).astype(np.uint8), cmap=None)
        # ax[3,4-3].axis('off')
        # ax[3,5-3].imshow(((pred[0].detach().cpu().numpy())*edges_and_texture[0]*255).astype(np.uint8), cmap=None)
        # ax[3,5-3].axis('off')
        # ax[3,6-3].imshow(((pred[0].detach().cpu().numpy())*edges_and_not_texture[0]*255).astype(np.uint8), cmap=None)
        # ax[3,6-3].axis('off')

        # ax[3,3-3].imshow(((pred_bin[0].detach().cpu().numpy())*sky_ground[0]*255).astype(np.uint8), cmap=None)
        # ax[3,3-3].axis('off')
        # ax[3,4-3].imshow(((pred_bin[0].detach().cpu().numpy())*trees[0]*255).astype(np.uint8), cmap=None)
        # ax[3,4-3].axis('off')
        # ax[3,5-3].imshow(((pred_bin[0].detach().cpu().numpy())*edges_and_texture[0]*255).astype(np.uint8), cmap=None)
        # ax[3,5-3].axis('off')
        # ax[3,6-3].imshow(((pred_bin[0].detach().cpu().numpy())*edges_and_not_texture[0]*255).astype(np.uint8), cmap=None)
        # ax[3,6-3].axis('off')

        # ax[4,3-3].imshow((((pred_bin[0].detach().cpu().numpy())*(1-sky_ground[0]))*255).astype(np.uint8), cmap=None)
        # ax[4,3-3].axis('off')
        # ax[4,4-3].imshow((((pred_bin[0].detach().cpu().numpy())*(1-trees[0]))*255).astype(np.uint8), cmap=None)
        # ax[4,4-3].axis('off')
        # ax[4,5-3].imshow((((pred_bin[0].detach().cpu().numpy())*(1-edges_and_texture[0]))*255).astype(np.uint8), cmap=None)
        # ax[4,5-3].axis('off')
        # ax[4,6-3].imshow((((pred_bin[0].detach().cpu().numpy())*(1-edges_and_not_texture[0]))*255).astype(np.uint8), cmap=None)
        # ax[4,6-3].axis('off')

        # ax[5,0].imshow(((normalized_substracted_pred[0])*255).astype(np.uint8), cmap=None)
        # ax[5,0].axis('off')

        # ax[5,1].imshow((((pred[0].detach().cpu().numpy())*(1-added_filters[0]))*255).astype(np.uint8), cmap=None)
        # ax[5,1].axis('off')

        # ax[5,2].imshow(((((pred[0].detach().cpu().numpy())*(1-added_filters[0]))*y[0].detach().cpu().numpy())*255).astype(np.uint8), cmap=None)
        # ax[5,2].axis('off')

        # ax[5,3].imshow(((normalized_substracted_pred[0]*y[0].detach().cpu().numpy())*255).astype(np.uint8), cmap=None)
        # ax[5,3].axis('off')

        # plt.show()

        positive = (pred > 0.1).int()

        # Ground truth masks
        gt_pos = (y > 0).int()   # ground truth positives
        gt_neg = (y == 0).int()  # ground truth negatives

        # True Positives: predicted 1 and GT = 1
        TP = ((positive == 1) & (gt_pos == 1)).sum()

        # False Positives: predicted 1 and GT = 0
        FP = ((positive == 1) & (gt_neg == 1)).sum()

        # False Negatives: predicted 0 and GT = 1
        FN = ((positive == 0) & (gt_pos == 1)).sum()

        # Precision, Recall, F1
        precision = TP / (TP + FP + 1e-8)
        recall    = TP / (TP + FN + 1e-8)
        f1        = 2 * (precision * recall) / (precision + recall + 1e-8)

        print("Precision:", precision.item())
        print("Recall:", recall.item())
        print("F1-score:", f1.item())
    
        # input('hipo')
        # positive_patches =torch.where(patched_score_chw>0,)[0]
        # negative_patches = torch.where(patched_score_chw==0,)[0].tolist()
            
        # # if positive_patches.shape[0]>0:
        #     negative_patches = random.choices(negative_patches,k=positive_patches.shape[0]*2)
        #     patches = torch.cat((patched_score_chw[positive_patches],patched_score_chw[negative_patches]),dim=0)
        #     patched_score_chw = patched_score_chw.float().to(device)
        #     pred = model(polynomial_patch)
        #     rectified_pred = torch.where(pred>0.5,1,0)
        #     loss = loss_fn((pred[patches].squeeze()),(patched_score_chw[patches].squeeze()))
        #     loss += -loss_mse(pred[positive_patches].mean(),pred[negative_patches].mean())
        #     loss += loss_mse(pred[positive_patches].mean(),torch.ones_like(pred[positive_patches]).mean())
        #     loss += loss_mse(pred[negative_patches].mean(),torch.zeros_like(pred[negative_patches]).mean())
        #     # print(patched_score_chw[positive_patches].mean(),pred[positive_patches].mean())
        #     # print(patched_score_chw[negative_patches].mean(),pred[negative_patches].mean())
        #     print("True positive:",((patched_score_chw[positive_patches]*rectified_pred[positive_patches]).sum())/positive_patches.shape[0])
        #     print("True negative:",((1-patched_score_chw[negative_patches])*(1-rectified_pred[negative_patches])).sum()/len(negative_patches))
        #     print('loss at iterate i ',i,' is ',np.round(loss.item(),4))
        #     # input('yipo')
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        
        
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_dir = 'weights'
    weights_path = os.path.join(weights_dir, 'SimpleModel.pt')
    os.makedirs(weights_dir, exist_ok=True)
    model = SimpleModel(in_channels=21,num_classes=1).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print(f"Loaded model weights from {weights_path}")
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
    for i in range(100):
        batch_image, batch_label = batched_import(batch_size=8)
        if not batch_label:
            continue
        numpy_batch_image = np.array(batch_image)
        numpy_processed_batch = apply_filters(numpy_batch_image)
        numpy_processed_batch = np.concatenate((numpy_processed_batch,numpy_batch_image.astype(float)/255),axis=3)
        processed_label_batch = binarize_objectness(batch_image,batch_label)
        train_loop(model,optimizer,numpy_processed_batch,processed_label_batch,device)

if __name__ == "__main__":
    main()