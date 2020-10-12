import os
import glob
import numpy as np

import matplotlib.pyplot as plt

x_list = np.arange(100) + 1

for csv_path in glob.glob('./experiments/model/*.csv'):
    model_name = os.path.basename(csv_path).replace('.csv', '')
    lines = open(csv_path, 'r').readlines()[1:]

    train_data_list = []
    validation_data_list = []

    for line in lines:
        data = line.strip().split(',')
        print(csv_path, data)

        try:
            epoch, phase, loss, accuracy = data
        except ValueError:
            epoch, phase, loss, accuracy = data[:-1]
        
        if phase == 'train':
            train_data_list.append([float(loss), float(accuracy)])
        else:
            validation_data_list.append([float(loss), float(accuracy)])

    train_data_list = np.asarray(train_data_list, dtype=np.float32)
    validation_data_list = np.asarray(validation_data_list, dtype=np.float32)

    plt.plot(x_list[:len(train_data_list)], train_data_list[:, 0], label=model_name + '_train')
    plt.plot(x_list[:len(validation_data_list)], validation_data_list[:, 0], label=model_name + '_validation')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.xlim(x_list[0], x_list[-1])

plt.title('# Summary')
plt.legend(loc='upper right')
# plt.savefig
plt.show()
