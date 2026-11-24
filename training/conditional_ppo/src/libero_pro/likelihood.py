"""Full path density represented in RLinf's chunk log-probability interface."""


def full_path_logprob(log_probs, action_chunk, action_dim):
    result = log_probs.new_zeros((log_probs.shape[0], action_chunk, action_dim))
    result[:, 0, 0] = log_probs.sum(dim=(1, 2, 3))
    return result
