import numpy as np
from .network import StaticMLP, forward_pass, backward_pass
from .pettingzoo_env import run_internal_episode

def pretrain_mlp(epochs=5000):
    print("Starting offline MLP pre-training...")
    # Initialize a fresh MLP
    phi_omega = StaticMLP(2, 5)
    target_object_pos = np.array([50.0, 50.0])

    for epoch in range(epochs):
        # 1. Generate a random synthetic observation
        drone_pos = np.random.rand(2) * 100.0
        obs_vector = target_object_pos - drone_pos
        dist_to_obj = np.linalg.norm(obs_vector)

        # 2. Establish the Supervised Ground Truth Rules
        if dist_to_obj < 20.0:
            ground_truth_target = np.array([0.05, 0.05, 0.05, 0.05, 0.80])
        else:
            # Bias
            dx, dy = obs_vector[0], obs_vector[1]

            if abs(dy) > abs(dx):
                if dy > 0:
                    # Target is significantly North
                    ground_truth_target = np.array([0.70, 0.05, 0.10, 0.10, 0.05])
                else:
                    # Target is significantly South
                    ground_truth_target = np.array([0.05, 0.70, 0.10, 0.10, 0.05])
            else:
                if dx > 0:
                    # Target is significantly East
                    ground_truth_target = np.array([0.10, 0.10, 0.70, 0.05, 0.05])
                else:
                    # Target is significantly West
                    ground_truth_target = np.array([0.10, 0.10, 0.05, 0.70, 0.05])

        # 3. Perform forward pass and calculate error
        measure_output, h1 = forward_pass(phi_omega, obs_vector)

        # Recompute h1 locally for the backward pass
        h1 = mlp_hidden_helper(phi_omega, obs_vector)

        # 4. Train the weights
        backward_pass(
            mlp=phi_omega, 
            obs=obs_vector, 
            h1=h1, 
            output_probs=measure_output, 
            target_measure=ground_truth_target, 
            learning_rate=0.02
        )
        
    print("Pre-training complete! Network weights are now frozen.")
    return phi_omega # Return the fully trained, frozen network

def mlp_hidden_helper(mlp, obs):
    # Helper to extract hidden layer 1 state for backpropagation.
    h1 = mlp.W1 @ obs + mlp.b1
    return np.maximum(0.0, h1)

def run_swarm_simulation(n_agents, steps, mode):
    phi_omega = pretrain_mlp(epochs=5000)
    print(
        f"Beginning PettingZoo simulation loop for {n_agents} agents over {steps} ticks..."
    )
    rollout = run_internal_episode(
        phi_omega=phi_omega,
        mode=mode,
        n_agents=n_agents,
        steps=steps,
        seed=7,
    )
    print("Simulation finished.")
    return rollout
    
