import os
import pytest
import pandas as pd
import numpy as np

from src.simulation.bandit_policy import UCB1Agent, PolicyEnvironment, run_policy_simulation

def test_ucb1_agent_initialization():
    agent = UCB1Agent(n_arms=3, c=1.0)
    assert agent.n_arms == 3
    assert agent.total_pulls == 0
    assert np.all(agent.counts == 0.0)
    assert np.all(agent.q_values == 0.0)

def test_ucb1_agent_initial_pulls():
    agent = UCB1Agent(n_arms=3, c=1.0)
    # The first 3 pulls must play each arm exactly once
    played = []
    for _ in range(3):
        arm = agent.select_arm()
        played.append(arm)
        agent.update(arm, 0.5)
        
    assert sorted(played) == [0, 1, 2]
    assert np.all(agent.counts == 1.0)
    assert agent.total_pulls == 3

def test_ucb1_agent_exploit():
    agent = UCB1Agent(n_arms=3, c=0.1)  # small c to encourage exploitation
    # Initialize
    for arm in range(3):
        agent.update(arm, 0.1)
        
    # Give Arm 1 a high reward
    agent.update(1, 0.9)
    
    # Next pull should exploit Arm 1
    next_arm = agent.select_arm()
    assert next_arm == 1

def test_policy_environment():
    env = PolicyEnvironment(random_state=42)
    
    # Test Arm 0
    reward, regret, is_shock = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(regret, float)
    assert isinstance(is_shock, bool)
    
    # Under supply shock state regret is computed relative to Arm 2 (expected 0.55)
    # Under normal state regret is computed relative to Arm 1 (expected 0.40)
    # For Arm 0 (expected 0.35):
    # - shock: regret = 0.55 - 0.35 = 0.20
    # - normal: regret = 0.40 - 0.35 = 0.05
    assert any(np.isclose(regret, val) for val in [0.20, 0.05])
    
    # Test optimal regret is 0 under respective shock states
    for _ in range(100):
        # Test Arm 2 (expected 0.55 under shock, 0.30 normal)
        # Optimal under shock is Arm 2 (0.55): regret should be 0.0
        # Optimal under normal is Arm 1 (0.40): regret should be 0.40 - 0.30 = 0.10
        _, reg2, shock2 = env.step(2)
        if shock2:
            assert np.isclose(reg2, 0.0)
        else:
            assert np.isclose(reg2, 0.10)
            
        # Test Arm 1 (expected 0.40)
        # Optimal under normal is Arm 1 (0.40): regret should be 0.0
        # Optimal under shock is Arm 2 (0.55): regret should be 0.55 - 0.40 = 0.15
        _, reg1, shock1 = env.step(1)
        if not shock1:
            assert np.isclose(reg1, 0.0)
        else:
            assert np.isclose(reg1, 0.15)

def test_run_policy_simulation(tmp_path):
    output_path = os.path.join(tmp_path, "ucb_results.csv")
    
    # Run short simulation of 50 steps
    df = run_policy_simulation(steps=50, output_csv_path=output_path, random_state=42)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert os.path.exists(output_path)
    
    # Verify columns
    expected_cols = [
        "step", "selected_arm", "arm_name", "is_supply_shock", "reward",
        "average_reward_arm_0", "average_reward_arm_1", "average_reward_arm_2",
        "pulls_arm_0", "pulls_arm_1", "pulls_arm_2", "cumulative_regret"
    ]
    for col in expected_cols:
        assert col in df.columns
