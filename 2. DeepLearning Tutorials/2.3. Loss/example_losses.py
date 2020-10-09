
import numpy as np

preds = np.asarray([
    # [0.3, 0.7],
    # [0.1, 0.9],
    [0.0, 1.0],
    # [0.6, 0.4],
], dtype=np.float32)

gts = np.asarray([
    # [0, 1],
    # [1, 0],
    [1, 0],
    # [1, 0]
], dtype=np.float32)

# accuracy = np.mean(np.argmax(preds, axis=1) == np.argmax(gts, axis=1))
# print(accuracy)

# L1 and L2 Loss
# L1_Loss = np.sum(np.abs(preds-gts)) / len(preds)
L1_Loss = np.mean(np.sum(np.abs(preds-gts), axis=1), axis=0)
L2_Loss = np.mean(np.sum((preds-gts)**2, axis=1), axis=0)

print(L1_Loss)
print(L2_Loss)

# Cross-Entropy
ce_loss = -1*np.mean(np.sum(gts*np.log(preds+1e-10)+(1-gts)*np.log(1-preds+1e-10),axis=1),axis=0)
print(ce_loss)

