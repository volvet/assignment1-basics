
import numpy as np
import torch

class DataLoader:
    def __init__(self, data, batch_size, content_length, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.content_length = content_length
        self.shuffle = shuffle

    def get_train_batch_data(self):
        idxs = np.random.randint(0, len(self.data) - self.content_length, size=self.batch_size)
        x = np.array([self.data[i:i + self.content_length] for i in idxs])
        y = np.array([self.data[i + 1:i + self.content_length + 1] for i in idxs])
        return torch.tensor(x), torch.tensor(y)

    def get_eval_batch_data(self):
        raise NotImplementedError("Evaluation batch data not implemented yet.")

    def __len__(self):
        return len(self.data) // self.batch_size


if __name__ == "__main__":
    # Example usage
    data = np.arange(100)
    loader = DataLoader(data, batch_size=32, content_length=7)
    x, y = loader.get_train_batch_data()
    print("Input batch (x):", x)
    print("Target batch (y):", y)