import torch
from torch.utils.data import Dataset
import inspect


class RandomGenerator(Dataset):
    """A random generator that returns randomly generated tensors

    Args:
        size (int):
            Size of the dataset (number of samples)
        tensors:
            A list of tuples defining tensor properties: shape, type, range
            All properties are optionals. Defaults are null, float, [0,1]
    """

    def __init__(self, size:int=256, tensors=[
                                            (int, 50, [0,99]), 
                                            (int, 50, [1, 1]), 
                                            (int, 50, [0,99])]):
        torch.manual_seed(0)
        self.size = size
        self.tensors = tensors

    def __len__(self):
        return self.size
    
    # def data_sample():

    #     return RandomGenerator()[0]

    def __getitem__(self, idx):
        """
        Returns:
            A tuple of tensors.
        """
        sample = []
        for properties in self.tensors:
            shape=[]
            dtype=float
            drange=[0,1]
            for property in properties:
                if type(property)==int:
                    shape.append(property)
                elif inspect.isclass(property):
                    dtype=property
                elif type(property) is list:
                    drange=property
            shape=tuple(shape)

            if 'int' in str(dtype):
                tensor=torch.randint(low=drange[0], high=drange[1]+1, size=shape, dtype=dtype)
            else:
                tensor=torch.rand(size=shape,dtype=dtype)*(drange[1]-drange[0])+drange[0]

            sample.append(tensor)

        return tuple(sample)


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    dataset = RandomGenerator()
    data = dataset[0]
    print(dataset.data_sample())


