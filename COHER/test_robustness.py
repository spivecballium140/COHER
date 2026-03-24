import os
import sys
import torch
import numpy as np
from CEDCS_Env import CEDCS_env
from agent_DRL.agents.actor_critic_agents.OTDPG import OTDPG
from agent_DRL.utilities.data_structures.Config import Config

def run_robustness_test():
    # 1. 基础配置
    config = Config()
    config.seed = 1
    config.use_GPU = torch.cuda.is_available()
    config.file_to_save_data_results = False
    config.use_server = 0
    config.visualise_individual_results = False
    config.visualise_overall_agent_results = False
    config.runs_per_agent = 1
    config.randomise_random_seed = True
    config.save_model = False
    
    # 2. 超参数 (保持与 main.py 一致)
    config.hyperparameters = {
        "Actor_Critic_Agents": {
            "learning_rate": 0.0001,
            "linear_hidden_units": [256, 128],
            "final_layer_activation": ["sigmoid"],
            "gradient_clipping_norm": 5.0,
            "discount_rate": 0.99,
            "epsilon_decay_rate_denominator": 1.0,
            "normalise_rewards": True,
            "exploration_worker_difference": 2.0,
            "clip_rewards": False,
            "HER_sample_proportion": 0.5,
            "Actor": {
                "learning_rate": 0.0001,
                "linear_hidden_units": [256, 128],
                "final_layer_activation": "sigmoid",
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },
            "Critic": {
                "learning_rate": 0.0001,
                "linear_hidden_units": [256, 128],
                "final_layer_activation": 'relu',
                "batch_norm": False,
                "buffer_size": 1000000,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },
            "min_steps_before_learning": 1000,
            "batch_size": 256,
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.25,
            "action_noise_std": 0.1,
            "action_noise_clipping_range": 0.25,
            "update_every_n_steps": 10,
            "learning_updates_per_learning_session": 1,
            "automatically_tune_entropy_hyperparameter": True,
            "entropy_term_weight": None,
            "add_extra_noise": False,
            "do_evaluation_iterations": True
        }
    }

    # 提取 Agent 组的超参数 (Fix for KeyError)
    config.hyperparameters = config.hyperparameters["Actor_Critic_Agents"]

    # 3. 设置测试场景
    # 注意：这里需要根据实际情况修改模型路径
    # 假设我们测试 200 个任务的场景 (Case 2)
    # 请确保您有对应的模型文件，例如：'./my_data_and_graph/models/CEDCS_200_seed1-OTDPG-final.pt'
    # 如果没有找到模型，将使用随机策略进行演示
    
    data_file = './instances/data_matrix_200_seed1.txt'
    # 尝试查找一个存在的模型文件 (您可以手动修改此路径)
    model_path = './my_data_and_graph/models/CEDCS_200_seed1-OTDPG-final.pt'
    
    # 鲁棒性设置：测试 10% 和 20% 的不可用率
    failure_rates = [0.1, 0.2]
    
    config.num_episodes_to_run = 100 # 测试 100 个 episode 以获取统计数据
    config.is_train = False # 关闭训练模式
    
    print("==================================================")
    print("开始鲁棒性测试 (Robustness Test)")
    print("==================================================")

    for rate in failure_rates:
        print(f"\n[Scenario] 制造服务不可用率 (Failure Rate): {rate * 100}%")
        
        # 初始化环境，传入 failure_rate
        # 参数顺序: file_path, CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum, Cnum, failure_rate
        config.environment = CEDCS_env(data_file, 200, 200, 5, 200, 300, 200, failure_rate=rate)
        
        # 加载模型
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            config.is_load = True
            config.load_path = model_path
        else:
            print(f"Warning: Model not found at {model_path}.")
            print("Running with RANDOM initialized agent (Results will be poor, please train first).")
            config.is_load = False
        
        # 初始化 Agent
        # OTDPG 是该项目中主要的 Agent
        agent = OTDPG(config)
        agent.actor_local.eval() # 设置为评估模式
        
        # 收集指标
        metrics = {
            'rewards': [],
            'time': [], 
            'cost': [], 
            'reliability': [], # 重点关注
            'energy': []
        }
        
        for i_epi in range(config.num_episodes_to_run):
            # 修改：使用 is_state_dict=True 以获取 goal，适配 HER 训练的模型输入要求
            state_dict = config.environment.reset(is_state_dict=True)
            state = state_dict['observation']
            goal = state_dict['desired_goal']
            
            done = False
            episode_reward = 0
            
            while not done:
                # 拼接 state 和 goal (9 + 1 = 10 维)
                state_with_goal = np.concatenate([state, goal])
                
                # 动作选择
                state_tensor = torch.FloatTensor(state_with_goal).unsqueeze(0).to(agent.device)
                action = agent.actor_local(state_tensor).cpu().data.numpy()
                action = action[0] 
                
                # Step 也要保持 is_state_dict=True
                next_state_dict, reward, done, _ = config.environment.step(action, is_state_dict=True)
                
                state = next_state_dict['observation']
                goal = next_state_dict['desired_goal']
                episode_reward += reward
            
            metrics['rewards'].append(episode_reward)
            
            metrics['rewards'].append(episode_reward)
            
            # 从环境获取最终的 QoS 指标
            # obtain_obj 返回 (weighted_obj, (f1...f8))
            _, details = config.environment.obtain_obj(return_details=True)
            
            metrics['time'].append(details[0])
            metrics['cost'].append(details[1])
            metrics['reliability'].append(details[2])
            metrics['energy'].append(details[7])
            
            if (i_epi + 1) % 10 == 0:
                print(f"Progress: {i_epi + 1}/{config.num_episodes_to_run} episodes done.")

        # 输出统计结果
        print(f"\n--- Results for {rate*100}% Failure Rate ---")
        print(f"Avg Reward:      {np.mean(metrics['rewards']):.4f}")
        print(f"Avg Reliability: {np.mean(metrics['reliability']):.4f} (Should decrease significantly)")
        print(f"Avg Makespan:    {np.mean(metrics['time']):.4f}")
        print(f"Avg Cost:        {np.mean(metrics['cost']):.4f}")
        print(f"Avg Energy:      {np.mean(metrics['energy']):.4f}")

if __name__ == "__main__":
    run_robustness_test()

