import torch
import torch.nn as nn
import math

class LeakyRNNCell(nn.Module):
    """
    A continuous-time RNN cell discretized via Euler's method.
    h_t = (1 - alpha) * h_{t-1} + alpha * f(W_in x_t + W_rec h_{t-1} + b) + noise
    where alpha = dt / tau
    """
    def __init__(self, input_size, hidden_size, dt, tau, sigma_rec=0.0, activation='softplus', w_rec_init='diag'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = dt / tau
        self.sigma = math.sqrt(2 / self.alpha) * sigma_rec
        self.w_rec_init = w_rec_init
        
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        
        if activation == 'softplus':
            self.activation = nn.Softplus()
            self._w_rec_start = 0.5
        elif activation == 'tanh':
            self.activation = torch.tanh
            self._w_rec_start = 1.0
        elif activation == 'relu':
            self.activation = torch.relu
            self._w_rec_start = 0.5
        elif activation == 'retanh':
            self.activation = lambda x: torch.tanh(torch.relu(x))
            self._w_rec_start = 0.5
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize input weights
        nn.init.normal_(self.weight_ih, mean=0.0, std=1.0 / math.sqrt(self.input_size))
        
        # Initialize recurrent weights
        if self.w_rec_init == 'diag':
            nn.init.eye_(self.weight_hh)
            self.weight_hh.data.mul_(self._w_rec_start)
        elif self.w_rec_init == 'randortho':
            nn.init.orthogonal_(self.weight_hh)
            self.weight_hh.data.mul_(self._w_rec_start)
        elif self.w_rec_init == 'randgauss':
            nn.init.normal_(self.weight_hh, mean=0.0, std=self._w_rec_start / math.sqrt(self.hidden_size))
        else:
            raise ValueError(f"Unknown w_rec_init: {self.w_rec_init}")
            
        # Initialize bias
        nn.init.zeros_(self.bias)
        
    def forward(self, input, hx=None):
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, device=input.device)
            
        # Linear transform: W_in * x + W_rec * h + b
        pre_activation = torch.matmul(input, self.weight_ih.t()) + torch.matmul(hx, self.weight_hh.t()) + self.bias
        
        # Apply non-linearity
        f_x = self.activation(pre_activation)
        
        # Add recurrent noise during training if sigma_rec > 0
        if self.training and self.sigma > 0:
            noise = torch.randn_like(hx) * self.sigma
            f_x = f_x + noise
            
        # Euler integration step
        h_next = (1 - self.alpha) * hx + self.alpha * f_x
        
        return h_next


class CognitiveRNN(nn.Module):
    """
    Full RNN model for cognitive tasks.
    Takes sequential inputs, returns outputs and optional hidden states for drift analysis.
    """
    def __init__(self, input_size, hidden_size, output_size, dt, tau, sigma_rec=0.0, activation='softplus', w_rec_init='diag'):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = LeakyRNNCell(input_size, hidden_size, dt, tau, sigma_rec, activation, w_rec_init)
        
        # Readout layer
        self.readout = nn.Linear(hidden_size, output_size)
        
        # Initialize readout weights
        nn.init.normal_(self.readout.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size))
        nn.init.zeros_(self.readout.bias)
        
    def forward(self, x, return_all_states=False):
        """
        Args:
            x: Tensor of shape (Seq_len, Batch, Input_size)
            return_all_states: Boolean, if True returns (outputs, states)
        """
        seq_len, batch_size, _ = x.size()
        hx = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        states = []
        outputs = []
        
        for t in range(seq_len):
            hx = self.rnn_cell(x[t], hx)
            states.append(hx)
            outputs.append(self.readout(hx))
            
        # Stack lists into tensors
        states_tensor = torch.stack(states)   # Shape: (Seq_len, Batch, Hidden_size)
        outputs_tensor = torch.stack(outputs) # Shape: (Seq_len, Batch, Output_size)
        
        if return_all_states:
            return outputs_tensor, states_tensor
            
        return outputs_tensor
