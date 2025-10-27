# Recurrent Neural Network (RNN) cell in PyTorch

This minimal PyTorch example implements a custom recurrent neural network (RNN) cell from first principles, showing how sequence memory emerges through feedback.

The cell maintains a hidden state vector h, which evolves over time using the current input x and the previous hidden state through the nonlinear update h = tanh(Wₕₕh + Wₓₕx). The output y = Wₕy h is then computed as a simple linear projection of the hidden state.

Unlike PyTorch’s built-in nn.RNN, this implementation makes every matrix and operation explicit, clearly illustrating how temporal dependencies are learned through recursive state updates rather than static input-output mappings.

  
code: [https://github.com/UkiDLucas/DNN-book](https://github.com/UkiDLucas/DNN-book/blob/main/code/RNN/Tiny_RNN.py)  
  
  
  
```

> import torch  
> import torch.nn as nn  
>   
> # pick device (use "mps" on Apple Silicon;  
> # macOS Metal hardware acceleration  
> device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  
>   
> class MyRNNCell(nn.Module):  
> 
> > # rnn_units: number of hidden neurons  
> > def __init__(self, rnn_units, input_dim, output_dim):  
> 
> > > super().__init__()
> > 
> > > # weight matrices, * 0.01 scales the random weight initialization to small values
> > 
> > > self.W_xh = nn.Parameter(torch.randn(rnn_units, input_dim) * 0.01)
> > 
> > > self.W_hh = nn.Parameter(torch.randn(rnn_units, rnn_units) * 0.01)
> > 
> > > self.W_hy = nn.Parameter(torch.randn(output_dim, rnn_units) * 0.01)
> > 
> > > # hidden state h initialized to zeros
> > 
> > > self.register_buffer("h", torch.zeros(rnn_units, 1))
> 
>   
> 
> > def forward(self, x):
> 
> > > # x is shape [input_dim, 1]
> > 
> > > self.h = torch.tanh(self.W_hh @ self.h + self.W_xh @ x)
> > 
> > > y = self.W_hy @ self.h
> > 
> > > return y, self.h
> > 
> > > # minimal usage example
> 
>   
>   
> if __name__ == "__main__":  
> 
> > rnn = MyRNNCell(rnn_units=16, input_dim=8, output_dim=4).to(device)
> > 
> > x_t = torch.randn(8, 1, device=device) # input vector at time t
> > 
> > y_t, h_t = rnn(x_t)
> > 
> > print(y_t.shape, h_t.shape) # torch.Size([4, 1]) torch.Size([16, 1])

```

  

A practical use of this minimal RNN cell is to **predict or generate sequential data**, where each step depends on the previous one.

For example:

- **Time series forecasting:** Feed in one value at a time (like daily temperatures or stock prices) and train it to predict the next value.
    
- **Character-level text generation:** Convert characters to one-hot vectors, feed them sequentially, and let the RNN learn to predict the next character.
    
- **Signal smoothing or sensor prediction:** Use it to process sequential readings (like a boat’s wind and wave sensors) to predict future conditions.
    

Even though it’s a tiny model, it demonstrates the whole idea: maintaining internal memory (`h`) to connect past inputs with future outputs.

Various RNN configurations:

- single input >  RNN cell > single result for binary classification
- many inputs > many RNN cells > single output: sentiment classification
- single input > many RNN cells > many outputs: text generation, image captions
- many inputs > many RNN cells > many outputs: translation, music generation

Think of an RNN as a storyteller that remembers what has already been said while deciding what comes next. Each RNN cell is like one frame in a film reel—receiving new input, updating its memory, and passing that memory forward. When you connect many cells in series, the network forms a chain of thought through time: it doesn’t see the whole story at once but recalls what just happened. In a simple one-to-one setup, it’s like hearing a single word and deciding “yes” or “no.” With many inputs feeding into a single output, it listens to a whole sentence before forming an opinion, such as judging sentiment. With one input producing many outputs, it’s as if the RNN takes one idea and tells a whole story, step by step. And when many inputs produce many outputs, it becomes a fluent translator or composer—listening, remembering, and responding continuously.

References:

- [https://github.com/UkiDLucas/DNN-book?](https://github.com/UkiDLucas/DNN-book?)
- [https://www.youtube.com/watch?v=GvezxUdLrEk&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI](https://www.youtube.com/watch?v=GvezxUdLrEk&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)