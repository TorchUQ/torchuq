import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import os
import torch
import numpy as np


def normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

def train_dataset(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform()
    ])
    
    train_dataset = datasets.ImageFolder(
        train_dir,
        train_transforms
    )
    
    return train_dataset
  
def val_dataset(data_dir):
    val_dir = os.path.join(data_dir, 'val')
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize_transform()
    ])
    
    val_dataset = datasets.ImageFolder(
        val_dir,
        val_transforms
    )
    
    return val_dataset
  
def data_loader(data_dir, batch_size=256, workers=2, pin_memory=True):
    train_ds = train_dataset(data_dir)
    val_ds = val_dataset(data_dir)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
in_hier = ImageNetHierarchy('/atlas/u/shengjia/data/imagenet/', '/atlas/u/shengjia/data/imagenet/info/')



def get_imagenet_classes_1000():
    reader = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'imagenet_wnids.txt'))
    return reader.readline().split()


# Utilities for imagenet
def get_imagenet_superclass(classes):
    """
    Input a list of imagenet wnids, and output a list of superclasses 
    
    Return:
    class_indices: a list of list of wnids, class_indices[i] is a subset of classes that are descendents of class_wnid[i]
    class_wnid: the wnid of the superclass 
    """
    wnid_to_index = {wnid:i for i, wnid in enumerate(classes)}
    
    # This computes the leaf nodes that are decendents of each node
    classes_stack = [item for item in classes]   
    num_leaf = {}    
    descendents = {item: {item} for item in classes}
    layer = {item:1 for item in classes}   # The depth of a node (distance to most distant leaf node)
    while len(classes_stack) != 0:
        wnid = classes_stack.pop()
        parent = in_hier.tree[wnid].parent_wnid

        if parent is None:
            continue
        if parent not in num_leaf:
            descendents[parent] = descendents[wnid] # This does not have to be a deep copy
            num_leaf[parent] = 1
            layer[parent] = layer[wnid] + 1
        else:
            descendents[parent] = descendents[parent] | descendents[wnid]
            num_leaf[parent] += 1
            layer[parent] = max(layer[parent], layer[wnid]+1)

        classes_stack.append(parent)
    
    class_wnids = []
    class_indices = []
    for class_id in descendents:
        parent = in_hier.tree[class_id].parent_wnid 

        # Only include nodes that have fewer descendents than its parent
        if parent is not None and len(descendents[class_id]) != len(descendents[parent]):
            class_wnids.append(class_id)
            indices = [wnid_to_index[wnid] for wnid in descendents[class_id]]
            class_indices.append(indices)
            
    return class_indices, class_wnids



def merge_classes(classes, num_elements, predictions, labels, verbose=False):
    """ Merge the predictions based on the imagenet tree 
    - classes: an array of size [num_classes] that contains strings with wordnet synset name
    - num_elements: an array of size [num_classes], the number of data samples in each class, the merge will try to produce as even classes as possible
    - predictions: an array of size [batch_size, num_classes]
    - labels: an int array of size [batch_size] where each element should be [0, num_classes)
    Returns: 
    - classes, num_elements, predictions, labels for the merged super-class
    """
    # Find the size of the smallest parent if merged
    
    wnid_to_index = {wnid:i for i, wnid in enumerate(classes)}
    
    success = False
    while not success:
        # This computes the number of leaf node that are decendents of each node
        classes_stack = [item for item in classes]   
        num_leaf = {}    
        layer = {item:1 for item in classes}   # The depth of a node (distance to most distant leaf node)
        while len(classes_stack) != 0:
            wnid = classes_stack.pop()
            parent = in_hier.tree[wnid].parent_wnid

            if parent is None:
                continue
            if parent not in num_leaf:
                num_leaf[parent] = 1
                layer[parent] = layer[wnid] + 1
            else:
                num_leaf[parent] += 1
                layer[parent] = max(layer[parent], layer[wnid]+1)
                
            classes_stack.append(parent)

        # For each leaf node whose parent is not a ancestor of another leaf node, merge the leaf with its parent
        success = True
        for i, wnid in enumerate(classes.copy()):
            parent = in_hier.tree[wnid].parent_wnid 
            if parent is not None and num_leaf[parent] == 1:
                classes[i] = parent
                wnid_to_index[parent] = wnid_to_index[wnid]
                success = False   # If any node has been merged, repeat this process
        
    # Compute the size of the hypothetical merges
    merged_size = {}
    merged_component = {}
    for i, wnid in enumerate(classes):
        parent = in_hier.tree[wnid].parent_wnid
        if parent is None:
            continue

        if parent not in merged_size:
            merged_size[parent] = num_elements[i]
            merged_component[parent] = [wnid]
        else:
            merged_size[parent] += 1
            merged_component[parent].append(wnid)

    min_key = None
    for key in merged_size:
        if ((min_key is None) or (merged_size[key] < merged_size[min_key])) and layer[key] == 2:  # Important: should only merge nodes with depth=2
            min_key = key
    
    if verbose:
        print("Remaining %d, Merging %s <--- %s size = %d" % (len(classes), in_hier.wnid_to_name[min_key], ' || '.join([in_hier.wnid_to_name[wnid] for wnid in merged_component[min_key]]), merged_size[min_key]))
    # print("Merging %s <--- %s size = %d" % (min_key, ' || '.join([wnid for wnid in merged_component[min_key]]), merged_size[min_key]))
    
    # Keep only the indices that have not been merged
    merged_index = [wnid_to_index[wnid] for wnid in merged_component[min_key]]
    keep_index = [i for i in range(len(classes)) if i not in merged_index]
    new_index = [i for i in range(len(classes)) if i not in merged_index or i == merged_index[0]]
    new_classes = classes[keep_index]
    new_num_elements = num_elements[keep_index]
    
    # Add the new merged class
    assert min_key not in new_classes, min_key + str(new_classes)
    new_classes = np.insert(new_classes, merged_index[0], min_key)   # The new superclass will take the index of the first subclass
    new_num_elements = np.insert(new_num_elements, merged_index[0], merged_size[min_key])
    
    # Modify the predictions to combine the probability/labels assigned to merged classes
    labels = F.one_hot(labels)    # Convert to one hot to treat predictions and labels in the same way 
    new_labels = labels[:, new_index]
    new_predictions = predictions[:, new_index]
    new_labels[:, merged_index[0]] = labels[:, merged_index].sum(dim=1)
    new_predictions[:, merged_index[0]] = predictions[:, merged_index].sum(dim=1)
    new_labels = torch.argmax(new_labels, dim=1)

    return new_classes, new_num_elements, new_predictions, new_labels


# val_ds = val_dataset('../data/imagenet/')
# classes = val_ds.classes
# num_elements = np.ones(1000, dtype=np.int)
# predictions = torch.ones(1000, 1000) / 1000.
# labels = torch.arange(1000)
# while len(classes) > 1:
#     classes, num_elements, predictions, labels = merge_classes(np.array(classes), num_elements, predictions, labels) 
#     # print(labels)
# #     plt.figure(figsize=(5, 5))
# #     plt.imshow(predictions[:20, :20])
# #     plt.show()