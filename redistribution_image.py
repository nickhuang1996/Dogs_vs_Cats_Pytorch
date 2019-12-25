import os
import shutil

def redistribution():
    data_file = os.listdir('../dogs_vs_cats/Dataset')
    dogs_file = list(filter(lambda x: x[:3] == 'dog', data_file))
    cats_file = list(filter(lambda x: x[:3] == 'cat', data_file))

    data_root = '../dogs_vs_cats/'
    train_root = '../dogs_vs_cats/train'
    val_root = '../dogs_vs_cats/val'
    for i in range(len(cats_file)):
        image_path = data_root + 'Dataset/' + cats_file[i]
        if i < len(dogs_file) * 0.9:
            new_path = train_root + '/cat/' + cats_file[i]
        else:
            new_path = val_root + '/cat/' + cats_file[i]
        shutil.move(image_path, new_path)

    for i in range(len(dogs_file)):
        image_path = data_root + 'Dataset/' + dogs_file[i]
        if i < len(dogs_file) * 0.9:
            new_path = train_root + '/dog/' + dogs_file[i]
        else:
            new_path = val_root + '/dog/' + dogs_file[i]
        shutil.move(image_path, new_path)

if __name__ == '__main__':
    redistribution()