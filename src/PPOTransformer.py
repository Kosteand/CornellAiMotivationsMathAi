import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torchutils
import torch.distributions as distributions
import torch.utils.data as data
import math
from dataclasses import dataclass
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out):
        super().__init__()

        self.Layer1 = nn.Linear(in_dim, hidden)
        self.Layer2 = nn.Linear(hidden, hidden)
        self.Layer3 = nn.Linear(hidden, out)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.activation(self.Layer1(x))
        x = self.activation(self.Layer2(x))
        x = self.Layer3(x)
        return x


class RolloutBuffer:
    def __init__(self):
        self.buffer_names = ["states",
                   "actions",
                   "rewards",
                   "log_probs",
                   "values",
                   "next_states",
                   "terminated",
                   "truncated",
                   "valid"
                   ]
        self.names_set = set(self.buffer_names)
        self.buffers = {k:[] for k in self.buffer_names}
        self._stale = False

    def append(self, experience: dict):
        if not self.names_set == set(experience): raise KeyError(f"key mismatch: {self.names_set.symmetric_difference(experience.keys())}")
        for arg, v in experience.items():
            self.buffers[arg].append(v)

    def __len__(self):
            return len(self.buffers["rewards"]) 
    
    def __getitem__(self, key):
        return np.array(self.buffers[key])

    def __repr__(self):
        return self.buffers.__repr__()
    

def compute_gaes(
    rewards: torch.Tensor,        # [T, N]
    values: torch.Tensor,         # [T, N]
    next_values: torch.Tensor,    # [T, N]
    terminated: torch.Tensor,     # [T, N]
    truncated: torch.Tensor,      # [T, N]
    
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ):
    # truncated is a stub for now
    """
    Compute GAE for vectorized environments.

    Args:
        rewards:      Tensor [T, N]
        values:       Tensor [T, N]
        next_values:  Tensor [T, N]
        terminated:   Tensor [T, N]
        truncated:    Tensor [T, N]

    Returns:
        advantages:   Tensor [T, N]
        returns:      Tensor [T, N]
    """
    T, N = rewards.shape
    done = torch.clamp(terminated + truncated, max=1.0) 
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(T)):
        # Bootstrap across truncation, but not true termination
        non_terminal_bootstrap = 1.0 - terminated[t]           # for the value bootstrap
        non_terminal_chain      = 1.0 - done[t]  

        delta = rewards[t] + gamma * next_values[t] * non_terminal_bootstrap - values[t]
        gae   = delta + gamma * gae_lambda * non_terminal_chain * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns

def EWMAUpdate(val, update, tau=0.01): 
    if val is not None:
        return val*(1-tau) + tau*update
    else:
        return update

@dataclass
class TrainStepResult:
    succeeded: bool
    actor_loss: float | None = None
    critic_loss: float | None = None
    entropy: float | None = None

    def update_ema_metrics(self, actor_loss, critic_loss, entropy, tau=0.01):
        self.actor_loss = EWMAUpdate(self.actor_loss, actor_loss, tau)
        self.critic_loss = EWMAUpdate(self.critic_loss, critic_loss, tau)
        self.entropy = EWMAUpdate(self.entropy, entropy, tau)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PPOAgent(nn.Module):
    def __init__(
            self, 
            obs_dim, 
            hidden_dim, 
            out, 
            device: torch.device,
            actor_lr=3e-4, 
            critic_lr=1e-3, 
            lr_decay=0.99, 
            n_steps=2048, 
            epochs=8, 
            batch_size=64,
            eps=0.2,
            ent_coeff=0.0,
            clipnorm = 1,
            gamma=0.99,
            gae_lambda=0.95,
            dtype: torch.dtype = torch.float64,
            ):
        super().__init__()

        #self.preprocessor = None

        # float64 throughout: the polynomial heatmap channels blow up like f**N, so
        # float32 loses the small-v resolution at higher N. The whole net + all rollout
        # tensors run in self.dtype so that precision survives end to end.
        self.dtype = dtype

        self.actor = MLP(obs_dim, hidden_dim, out)
        self.critic = MLP(obs_dim, hidden_dim, 1)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optim, gamma=lr_decay)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_optim, gamma=lr_decay)

        self.n_steps = n_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.eps = eps
        self.ent_coeff = ent_coeff
        self.clipnorm = clipnorm
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.device = device
        self.to(device=device, dtype=self.dtype)   # cast all params to self.dtype + move


    def forward(self, x):
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        logits = self.actor(x)
        values = self.critic(x)
        return logits, values.squeeze(-1)
    
    def take_action(self, x):
        with torch.inference_mode():
            logits, values = self.forward(x)
            action_dist = distributions.Categorical(logits=logits)
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions)
        return actions, log_probs, values
    
    def train_step(
                self, 
                rollout_buffer:RolloutBuffer, 
                ):
        if rollout_buffer._stale: raise ValueError("rollout buffer stale")
        if len(rollout_buffer) < self.n_steps: return TrainStepResult(False)

        train_step_result = TrainStepResult(False)
        state_tensor = torch.as_tensor(rollout_buffer["states"], dtype=self.dtype, device=self.device)          # [T, N, obs_dim]
        action_tensor = torch.as_tensor(rollout_buffer["actions"], dtype=torch.int64, device=self.device)          # [T, N]
        rewards_tensor = torch.as_tensor(rollout_buffer["rewards"], dtype=self.dtype, device=self.device)       # [T, N]
        log_prob_tensor = torch.as_tensor(rollout_buffer["log_probs"], dtype=self.dtype, device=self.device)    # [T, N]
        values_tensor = torch.as_tensor(rollout_buffer["values"], dtype=self.dtype, device=self.device)         # [T, N]
        next_states_tensor = torch.as_tensor(rollout_buffer["next_states"], dtype=self.dtype, device=self.device)  # [T, N, obs_dim]
        terminated_tensor = torch.as_tensor(rollout_buffer["terminated"], dtype=self.dtype, device=self.device) # [T, N]
        truncated_tensor = torch.as_tensor(rollout_buffer["truncated"], dtype=self.dtype, device=self.device)   # [T, N]

        
        with torch.no_grad():
            # value_net should output [T, N, 1] or [T, N]
            next_values_tensor = self.critic(next_states_tensor).squeeze(-1)  # [T, N]

        advantage_tensor, returns_tensor = compute_gaes(
            rewards=rewards_tensor,
            values=values_tensor,
            next_values=next_values_tensor,
            terminated=terminated_tensor,
            truncated=truncated_tensor,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        T, N = rewards_tensor.shape
        obs_dim = state_tensor.shape[-1] 

        state_tensor = state_tensor.reshape(T * N, obs_dim)
        action_tensor = action_tensor.reshape(T * N)
        log_prob_tensor = log_prob_tensor.reshape(T * N)
        values_tensor = values_tensor.reshape(T * N)
        advantage_tensor = advantage_tensor.reshape(T * N)
        returns_tensor = returns_tensor.reshape(T * N)


    #Prevents dead steps when truncating or terminating
        valid_tensor = torch.as_tensor(
            rollout_buffer["valid"], dtype=self.dtype, device=self.device
        ).reshape(T * N).bool()
        state_tensor     = state_tensor[valid_tensor]
        action_tensor    = action_tensor[valid_tensor]
        log_prob_tensor  = log_prob_tensor[valid_tensor]
        values_tensor    = values_tensor[valid_tensor]      # ← don't forget this one
        advantage_tensor = advantage_tensor[valid_tensor]
        returns_tensor   = returns_tensor[valid_tensor]

        advantage_tensor = (advantage_tensor-advantage_tensor.mean()) / advantage_tensor.std().clamp_min(1e-4)

        n_valid = state_tensor.shape[0]

        for epoch in range(self.epochs):
            indices = torch.randperm(n_valid)
            for batch in range(0, n_valid, self.batch_size):
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()

                batch_indices = indices[batch:batch+self.batch_size]
                batch_states = state_tensor[batch_indices]
                batch_actions = action_tensor[batch_indices]
                batch_log_probs = log_prob_tensor[batch_indices]
                #batch_values = values_tensor[batch_indices] # value function clipping usually hurts
                batch_advantages = advantage_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                new_logits, new_values = self(batch_states)

                new_action_dist = distributions.Categorical(logits=new_logits)
                new_log_probs = new_action_dist.log_prob(batch_actions)

                ratio = torch.exp(new_log_probs-batch_log_probs)

                surr1 = ratio*batch_advantages
                surr2 = torch.clip(ratio, min=1-self.eps, max=1+self.eps)*batch_advantages

                actor_loss = torch.mean(-torch.minimum(surr1, surr2))
                
                entropy = new_action_dist.entropy()
                entropy_loss = torch.mean(-(entropy*self.ent_coeff))

                critic_loss = torch.square(batch_returns-new_values).mean()

                total_loss = actor_loss + entropy_loss + 0.5 * critic_loss

                total_loss.backward()

                torchutils.clip_grad_norm_(self.critic.parameters(), self.clipnorm)
                torchutils.clip_grad_norm_(self.actor.parameters(), self.clipnorm)

                self.actor_optim.step()
                self.critic_optim.step()

                train_step_result.update_ema_metrics(actor_loss.item(), critic_loss.item(), entropy.mean().item())
        
        rollout_buffer._stale = True
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()
        train_step_result.succeeded = True
        return train_step_result 


        





        
        

def main():
    buff = RolloutBuffer()
    buff.append({'states':1, 'truncated':1, 'values':1, 'log_probs':1, 'terminated':1, 'next_states':1, 'actions':1, 'rewards':1})
    breakpoint()


if __name__ == "__main__":
    main()
    