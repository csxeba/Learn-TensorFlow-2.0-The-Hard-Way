import numpy as np


def discount_rewards(rewards, dones, discount_factor):
    discounted = np.empty_like(rewards)
    cumulative_sum = 0.
    for i in range(len(rewards)-1, -1, -1):
        cumulative_sum *= (1 - dones[i]) * discount_factor
        cumulative_sum += rewards[i]
        discounted[i] = cumulative_sum
    return discounted


