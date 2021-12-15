import numpy as np
from utils import plot_images

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import numpy as np
def get_train_valid_loader(
    data_dir,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,

    show_sample=False,
    num_workers=4,
    pin_memory=False,
):



    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load dataset
    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=trans)

    num_train = len(dataset)
    split = int(np.floor(valid_size * num_train))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [num_train - split,split])

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=9,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir, batch_size, num_workers=4, pin_memory=False):
    """Test datalaoder"""

    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load dataset
    dataset = datasets.MNIST(data_dir, train=False, download=True, transform=trans)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader




class MNISTClutteredDataset(Dataset):
  def __init__(self, array,array_orig,label):
    self.array = array
    self.label = label
    self.array_orig = array_orig
    self.orig_size = (40,40)

  def __len__(self):
    return len(self.array)

  def __getitem__(self, i):
    img = torch.tensor(self.array[i]).reshape((40,40)).unsqueeze(0).unsqueeze(0)
    img_orig = torch.tensor(self.array_orig[i]).reshape((28, 28)).unsqueeze(0).unsqueeze(0)
    return img.squeeze(0),img_orig.squeeze(0),torch.tensor(self.label[i][0],dtype=torch.int64)


def collate_fn(batch):
  image_list =[]
  label_list = []
  image_orig_list = []
  for images,image_orig,labels in batch:
    image_list.append(images)
    label_list.append(labels)
    image_orig_list.append(image_orig)

  return (torch.stack(image_list,dim = 0),torch.stack(image_orig_list,dim = 0)), torch.stack(label_list,dim=0).type(torch.int64)


def get_train_valid_loader_mnist_clut(
    data_dir,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    show_sample=False,
    num_workers=4,
    pin_memory=False,
):


    X_train = np.load(data_dir+'trainxclut40.npy')
    X_train = X_train.reshape((X_train.shape[0]*X_train.shape[1],-1))
    y_train = np.load(data_dir+'trainy.npy')
    y_train = y_train.reshape((y_train.shape[0] * y_train.shape[1], -1))
    X_valid = np.load(data_dir+'valxclut40.npy')
    X_valid = X_valid.reshape((X_valid.shape[0] * X_valid.shape[1], -1))
    y_valid = np.load(data_dir+'valy.npy')
    y_valid = y_valid.reshape((y_valid.shape[0] * y_valid.shape[1], -1))
    X_train_mnist = np.load(data_dir+'trainx.npy')
    X_train_mnist = X_train_mnist.reshape((X_train_mnist.shape[0] * X_train_mnist.shape[1], -1))
    X_val_mnist = np.load(data_dir+'valx.npy')
    X_val_mnist = X_val_mnist.reshape((X_val_mnist.shape[0] * X_val_mnist.shape[1], -1))
    train_ds_mc = MNISTClutteredDataset(X_train,X_train_mnist,y_train)
    val_ds_mc = MNISTClutteredDataset(X_valid,X_val_mnist,y_valid)

    mnist_clut_train_loader = torch.utils.data.DataLoader(train_ds_mc,batch_size=batch_size,collate_fn = collate_fn,\
                                                          shuffle=shuffle,num_workers=num_workers,pin_memory=pin_memory)
    mnist_clut_val_loader = torch.utils.data.DataLoader(val_ds_mc,batch_size=batch_size,collate_fn = collate_fn,\
                                                        shuffle=False,num_workers=num_workers,pin_memory=pin_memory)

    return (mnist_clut_train_loader,mnist_clut_val_loader)

def get_test_loader_mnist_clut(data_dir, batch_size, num_workers=4, pin_memory=False):
    X_test = np.load(data_dir+'testxclut40.npy')
    y_test = np.load(data_dir+'testy.npy')
    X_test_mnist = np.load(data_dir+'testx.npy')

    test_ds_mc = MNISTClutteredDataset(X_test,X_test_mnist,y_test)
    mnist_clut_test_loader = torch.utils.data.DataLoader(test_ds_mc,batch_size=batch_size,collate_fn = collate_fn,\
                                                         shuffle=True,num_workers=num_workers,pin_memory=pin_memory)
    return mnist_clut_test_loader