import torch as t

class TDNN(t.nn.Module):
    def __init__(self, kernel, hidden_dim, taps):
        super(TDNN, self).__init__()
        self.kernel = kernel
        self.hidden_dim = hidden_dim

        conv_size = (taps+1) - self.kernel + 1
        self.conv1d = t.nn.Conv1d(1,1,self.kernel)
        self.relu = t.nn.ReLU()
        self.hidden = t.nn.Linear(conv_size, 1)
        
    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.hidden(x)
        return x
