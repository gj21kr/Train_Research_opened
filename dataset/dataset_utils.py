import matplotlib.pyplot as plt
import numpy.ma as ma

def visualize(image, gt, num_frame=None, axis=3):
    if num_frame is None:
        num_frame = image.shape[axis-1] //2
    if axis==1:
        image_2d = image[num_frame]
        gt_2d = gt[num_frame]
    elif axis==2:
        image_2d = image[:,num_frame]
        gt_2d = gt[:,num_frame]
    elif axis==3:
        image_2d = image[:,:,num_frame]
        gt_2d = gt[:,:,num_frame]

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(image_2d,'gray')
    ax1.axis('off')
    ax2.imshow(gt_2d,'gray')
    ax2.axis('off')
    image_masked = ma.masked_array(gt_2d == 0, gt_2d)
    ax3.imshow(image_2d, 'gray')
    ax3.imshow(image_masked, alpha=0.3, cmap='hot')
    ax3.axis('off')
    plt.show()
