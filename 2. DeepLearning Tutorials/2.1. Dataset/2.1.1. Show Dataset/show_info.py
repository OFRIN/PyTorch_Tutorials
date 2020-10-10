import os

root_dir = '../../Toy_Dataset/'

cat_image_names = os.listdir(root_dir + 'cat/')
dog_image_names = os.listdir(root_dir + 'dog/')

print(f'# cat : {len(cat_image_names)}')
print(f'# dog : {len(dog_image_names)}')

