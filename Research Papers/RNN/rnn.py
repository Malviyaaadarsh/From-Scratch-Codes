"""
Implementation of the Recurrent Neural Network (Vanilla RNN) as described in paper "Finding Structures in Time" by Elman.
Steps to implement the RNN:
1. Implementaion of Single RNN Cell 
2. Initialzing Hidden State for the RNN
3. Implementation of Forward Pass through Sequence of RNN Cells
4. Implementation of Backward Pass or Backpropagation through time (BPTT) for RNN
5. Simulation of Vanishing Gradient Problem in RNN
6. Implementation of the RNN Class which combines all the above steps and provides a complete RNN model
"""

import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Implementation of a single RNN cell. Forward pass through a single RNN cell."""
    h_t = np.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b_h)
    return h_t

def hidden_init(batch_size: int, hidden_dim: int) -> np.ndarray:
    """
    Initializes the hidden state for the RNN.
    """
    return np.zeros((batch_size, hidden_dim))

def forward_rnn(X: np.ndarray, h_0: np.ndarray,W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through a sequence of RNN cells.
    """
    batch,T,input_dim = X.shape 
    h_t = h_0 
    hidden_dim = h_0.shape[1]
    hidden_states = np.zeros((batch,T,hidden_dim))
    for t in range(T):
        X_t = X[:,t,:]
        h_t = np.tanh(X_t @ W_xh.T + h_t @ W_hh.T + b_h)
        hidden_states[:,t,:] = h_t
    h_final = h_t
    return hidden_states, h_final

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backpropagation through time (BPTT) for a single RNN time step.
    """
    dtanh = (1 - h_t ** 2) * dh_next
    dW_hh = np.dot(dtanh.T,h_prev)
    dh_prev = np.dot(dtanh,W_hh)
    return (dh_prev,dW_hh)

def gradient_norm_decay(T: int, W_hh: np.ndarray)->list:
    """
    Simulates the vanishing gradient problem in RNNs.
    """
    spectral_norm = np.linalg.norm(W_hh,ord=2)
    gradient_norms = [1.0]
    for t in range(1,T):
        gradient_norms.append(gradient_norms[-1] * spectral_norm)
    return gradient_norms 


class VanillaRNN:
    """
    Implementation of the Vanilla RNN model.
    """
    def __init__(self, input_dim:int, hidden_dim:int,output_dim:int):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Initialize weights and biases by Xavier initialization (scale = sqrt(2/(fan_in + fan_out)))
        self.W_xh = np.random.randn(hidden_dim,input_dim) * np.sqrt(2.0/(input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim,hidden_dim) * np.sqrt(2.0/(2*hidden_dim))
        self.W_hy = np.random.randn(output_dim,hidden_dim) * np.sqrt(2.0/(hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X:np.ndarray,h_0:np.array=None) -> tuple:
        """
        Forward pass through entire sequence of Cells.
        """
        batch,T,input_dim = X.shape
        hidden_dim = self.hidden_dim
        output_dim = self.output_dim
        hidden_states = np.zeros((batch,T,hidden_dim))
        y_seq = np.zeros((batch,T,output_dim))
        if h_0 is None:
            h_t = np.zeros((batch,hidden_dim))
        else:
            h_t = h_0
        for t in range(T):
            X_t = X[:,t,:]
            h_t = np.tanh(X_t @ self.W_xh.T + h_t @ self.W_hh.T + self.b_h)
            y_t = h_t @ self.W_hy.T + self.b_y
            hidden_states[:,t,:] = h_t
            y_seq[:,t,:] = y_t
        h_final = h_t
        return y_seq, h_final
