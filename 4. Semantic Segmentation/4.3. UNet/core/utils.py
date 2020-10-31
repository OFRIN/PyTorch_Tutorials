
def get_numpy_from_tensor(tensor):
    return tensor.cpu().detach().numpy()