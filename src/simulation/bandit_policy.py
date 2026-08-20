import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("bandit_policy")

ARM_NAMES = {
    0: "Price Ceiling Only",
    1: "Price Ceiling + Subsidy",
    2: "Price Ceiling + Import Waiver"
}

class UCB1Agent:
    """
    Upper Confidence Bound (UCB1) Multi-Armed Bandit decision agent.
    """
    def __init__(self, n_arms: int = 3, c: float = 1.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.q_values = np.zeros(n_arms)
        self.total_pulls = 0

    def select_arm(self) -> int:
        # 1. Play unpulled arms first to initialize average reward estimates
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
                
        # 2. Apply UCB1 formula: select arm maximizing average_reward + exploration_bound
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            exploration_bound = self.c * np.sqrt(np.log(self.total_pulls) / self.counts[arm])
            ucb_values[arm] = self.q_values[arm] + exploration_bound
            
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.total_pulls += 1
        # Incremental average update
        self.q_values[arm] += (reward - self.q_values[arm]) / self.counts[arm]

class PolicyEnvironment:
    """
    Policy environment stochastically generating pricing shocks and evaluating reward payoffs
    for price ceiling, consumer subsidy, and import duty waiver arms.
    """
    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)

    def step(self, action: int) -> Tuple[float, float, bool]:
        """
        Executes selected policy intervention.
        Returns:
          - reward (float): stabilization benefit minus implementation cost
          - regret (float): optimal expected reward minus chosen arm expected reward
          - is_supply_shock (bool): active state shock indicator
        """
        # 1. Stochastic supply shock (20% probability)
        is_supply_shock = self.rng.rand() < 0.20
        
        # 2. Determine stabilization payoff and costs
        if action == 0:
            # Price ceiling only: Moderate effect, low cost
            stabilization = self.rng.normal(0.40, 0.05)
            cost = 0.05
            expected_reward = 0.35
        elif action == 1:
            # Price ceiling + Subsidy: High effect, high cost
            stabilization = self.rng.normal(0.75, 0.05)
            cost = 0.35
            expected_reward = 0.40
        elif action == 2:
            # Price ceiling + Import Waiver: Extremely high under shock, moderate cost
            if is_supply_shock:
                stabilization = self.rng.normal(0.70, 0.08)
                expected_reward = 0.55
            else:
                stabilization = self.rng.normal(0.45, 0.05)
                expected_reward = 0.30
            cost = 0.15
        else:
            raise ValueError(f"Invalid intervention arm index: {action}")
            
        reward = float(stabilization - cost)
        
        # 3. Calculate optimal expected reward & regret
        if is_supply_shock:
            # Arm 2 (Import waiver) is optimal under shock: expected 0.55
            optimal_expected = 0.55
        else:
            # Arm 1 (Subsidy) is optimal under normal conditions: expected 0.40
            optimal_expected = 0.40
            
        regret = float(optimal_expected - expected_reward)
        
        return reward, regret, is_supply_shock

def run_policy_simulation(
    steps: int = 1000,
    output_csv_path: str = "data/simulations/ucb_results.csv",
    random_state: int = 42
) -> pd.DataFrame:
    logger.info("Starting Multi-Armed Bandit Policy Simulation (steps=%d)...", steps)
    
    agent = UCB1Agent(n_arms=3, c=1.0)
    env = PolicyEnvironment(random_state=random_state)
    
    history: List[Dict[str, Any]] = []
    cumulative_regret = 0.0
    
    for t in range(1, steps + 1):
        # Action choice
        arm = agent.select_arm()
        
        # Take step
        reward, regret, is_shock = env.step(arm)
        
        # Update agent
        agent.update(arm, reward)
        
        # Accumulate regret
        cumulative_regret += regret
        
        history.append({
            "step": t,
            "selected_arm": arm,
            "arm_name": ARM_NAMES[arm],
            "is_supply_shock": int(is_shock),
            "reward": float(round(reward, 4)),
            "average_reward_arm_0": float(round(agent.q_values[0], 4)),
            "average_reward_arm_1": float(round(agent.q_values[1], 4)),
            "average_reward_arm_2": float(round(agent.q_values[2], 4)),
            "pulls_arm_0": int(agent.counts[0]),
            "pulls_arm_1": int(agent.counts[1]),
            "pulls_arm_2": int(agent.counts[2]),
            "cumulative_regret": float(round(cumulative_regret, 4))
        })
        
    df_history = pd.DataFrame(history)
    
    # Save CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_history.to_csv(output_csv_path, index=False)
    logger.info("Saved policy convergence logs successfully to %s", output_csv_path)
    
    # Summarize final counts and estimated values
    logger.info("=== UCB1 Multi-Armed Bandit Convergence Summary ===")
    for arm in range(3):
        logger.info("Intervention Arm %d [%s]: Pulls = %d, Estimated Mean Reward = %s",
            arm, ARM_NAMES[arm], agent.counts[arm], round(agent.q_values[arm], 4)
        )
    logger.info("Final Cumulative Regret: %s", round(cumulative_regret, 4))
    
    return df_history

def main():
    parser = argparse.ArgumentParser(description="Multi-Armed Bandit Simulation CLI")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulated steps")
    parser.add_argument("--output", type=str, default="data/simulations/ucb_results.csv", help="Path to save CSV logs")
    args = parser.parse_args()
    
    run_policy_simulation(args.steps, args.output)

if __name__ == "__main__":
    main()
