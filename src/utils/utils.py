def split_data(train_data, validation_split=.2):
    num_train = int(round((1 - validation_split) * len(train_data[0])))
    train_images, train_classes = train_data
    train_images = train_images[:num_train]
    train_classes = train_classes[:num_train]
    validation_images = train_data[num_train:]
    validation_classes = train_data[num_train:]
    train_data = (train_images, train_classes)
    validation_data = (validation_images, validation_classes)
    return train_data, validation_data

def get_class_names(dataset_name='VOC2007'):
    if dataset_name == 'VOC2007':
        class_names = ['background','aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person',
                       'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    else:
        raise Exception('Invalid %s dataset' % dataset_name)
    return class_names

def scheduler(epoch, decay=0.9, base_learning_rate=3e-4):
    return base_learning_rate * decay**(epoch)

def get_arg_to_class(class_names):
    return dict(zip(list(range(len(class_names))), class_names))
