import torch
import torch.nn as nn

# pick device (use "mps" on Apple Silicon if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class MyRNNCell(nn.Module):
    def __init__(self, rnn_units, input_dim, output_dim):
        super().__init__()
        # weight matrices
        self.W_xh = nn.Parameter(torch.randn(rnn_units, input_dim) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(rnn_units, rnn_units) * 0.01)
        self.W_hy = nn.Parameter(torch.randn(output_dim, rnn_units) * 0.01)
        # hidden state h initialized to zeros
        self.register_buffer("h", torch.zeros(rnn_units, 1))

    def forward(self, x):
        # x is shape [input_dim, 1]
        self.h = torch.tanh(self.W_hh @ self.h + self.W_xh @ x)
        y = self.W_hy @ self.h
        return y, self.h

# minimal usage example
if __name__ == "__main__":
    rnn = MyRNNCell(rnn_units=16, input_dim=8, output_dim=4).to(device)
    x_t = torch.randn(8, 1, device=device)  # input vector at time t
    y_t, h_t = rnn(x_t)
    print(y_t.shape, h_t.shape)  # torch.Size([4, 1]) torch.Size([16, 1])

# expected: 
# DNN-book/code/RNN/Tiny_RNN.py"
# torch.Size([4, 1]) torch.Size([16, 1])