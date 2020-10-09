import numpy as np
import matplotlib.pyplot as plt

lines = open('log.txt', 'r').readlines()

train_accuracy_list = []
valid_accuracy_list = []

for line in lines:
    # print(line.strip().split(','))
    splitlog =line.strip().split(',')
    
    name, percent = splitlog[2].strip().split('=')

    if 'train' in name:
        train_accuracy_list.append(float(percent[:-1]))
    else:
        valid_accuracy_list.append(float(percent[:-1]))

print(train_accuracy_list)
print(valid_accuracy_list)

max_epoch = len(train_accuracy_list)
epochs = np.arange(max_epoch)+1

plt.plot(epochs,train_accuracy_list, label='Train')
plt.plot(epochs,valid_accuracy_list, label='Validation')

plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.title('Classification')
plt.legend(loc ='lower right')

# plt.show()
plt.savefig(fname='log.png')

# X = Epoch, Y = Accuracy
# plt.plot([0, 1, 2], [100, 200, 300])
# plt.show()