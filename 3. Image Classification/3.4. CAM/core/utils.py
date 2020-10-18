import numpy as np

def get_numpy_from_tensor(tensor):
    return tensor.cpu().detach().numpy()

def denormalize(image, mean, std):
    mean = np.asarray(mean, dtype=np.float32).reshape((3, 1, 1))
    std = np.asarray(std, dtype=np.float32).reshape((3, 1, 1))
    
    image = image * std + mean
    image = image * 255

    image = image.transpose((1, 2, 0)) # [224, 224, 3]
    image = image.astype(np.uint8)[:, :, ::-1]

    return image

def get_cam(features, weight, bias):
    cam = weight[:, np.newaxis, np.newaxis] * features + np.reshape(bias, (1, 1, 1))
    cam = np.sum(cam,axis=0)
    
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = cam * 255

    return cam.astype(np.uint8)