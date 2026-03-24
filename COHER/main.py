from CEDCS_Env import CEDCS_env


from agent_DRL.agents.actor_critic_agents.OTDPG import OTDPG
from agent_DRL.agents.actor_critic_agents.DDPG import DDPG
from agent_DRL.agents.actor_critic_agents.TD3 import TD3
from agent_DRL.agents.actor_critic_agents.TD3_OTD import TD3_OTD
from agent_DRL.agents.actor_critic_agents.TD3_HER import TD3_HER
from agent_DRL.agents.actor_critic_agents.DDPG_HER import DDPG_HER
from agent_DRL.agents.actor_critic_agents.DDPG_OTD import DDPG_OTD
from agent_DRL.agents.actor_critic_agents.SAC import SAC
from agent_DRL.agents.actor_critic_agents.A2C import A2C
from agent_DRL.agents.actor_critic_agents.PPO import PPO


from agent_DRL.utilities.data_structures.Config import Config
from agent_DRL.agents.Trainer import Trainer

import torch
torch.set_num_threads(1)


config = Config()
config.seed = 1
config.use_GPU = True
embedding_dimensions = []

config.file_to_save_data_results = False    #是否保存为pkl文件
config.use_server = 0

config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = False
config.standard_deviation_results = 1.0     #画均值方差图时，阴影区域的范围是几倍的标准差

config.runs_per_agent = 1   
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False    


config.hyperparameters = {

    "Actor_Critic_Agents": {

        "learning_rate": 0.0001,
        "linear_hidden_units": [256, 128],  # 增大网络：从 [64, 16] -> [256, 128] 以适应复杂的 QoS 状态空间
        "final_layer_activation": ["sigmoid"],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,
        "HER_sample_proportion": 0.5, # 降低 HER 比例：0.8 -> 0.5，减少偏差

        "Actor": {
            "learning_rate": 0.0001,
            "linear_hidden_units": [256, 128], # 增大 Actor 网络
            "final_layer_activation": "sigmoid",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0001,
            "linear_hidden_units": [256, 128], # 增大 Critic 网络
            "final_layer_activation": 'relu',
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 1000,
        "batch_size": 256, # 调整 Batch Size：从 1024 -> 256，加快更新频率和泛化能力
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.1,  # 降低动作噪声：0.2 -> 0.1，利于后期收敛
        "action_noise_clipping_range": 0.25,  # 0.5 -> 0.25
        "update_every_n_steps": 10, # 加快更新频率：从 20 -> 10
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    },

    "Policy_Gradient_Agents": {
        "learning_rate": 0.0005,    
        "linear_hidden_units": [64, 16],   #[64,16]
        "final_layer_activation": "sigmoid",  #sigmoid
        "learning_iterations_per_round": 1,  #1
        "discount_rate": 0.99,
        "batch_norm": False,
        "clip_epsilon": 0.1,       #0.1
        "episodes_per_learning_round": 1,  # number of CPU cores for PPO parallelly
        "normalise_rewards": True,
        "gradient_clipping_norm": 7.0,     #7.0
        "mu": 0.0,  # only required for continuous action games
        "theta": 0.0,  # only required for continuous action games.0.0
        "sigma": 0.0,  # only required for continuous action games  0.0
        "epsilon_decay_rate_denominator": 1.0,
        "clip_rewards": False,
        "batch_size": 256,
        "learning_updates_per_learning_session": 10,
        
        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "sigmoid",
            "batch_norm": False,
            "gradient_clipping_norm": 5.0,
        },
        "Critic": {
            "learning_rate": 0.001,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "gradient_clipping_norm": 5.0,
        }
    },
}

import os

if __name__ == '__main__':

    # env = CEDCS_env('data_matrix_200.txt', 200, 200, 5, 200, 300, 200)
    #调节训练还是验证
    config.is_train = False  # 设置为 False 进行测试/验证
    config.is_load = True    # 设置为 True 以加载模型
    config.save_model = False  # 验证时不需要保存模型

    config.load_path_epix = None  # trained model of which episodes 3000:1118-trained-model

    config.num_episodes_to_run = 10  # 验证时只需运行少量 episodes，例如 100
    config.save_fre = 200
    config.record_interval = 500  # every which epis record the transitions
    
    use_curriculum = False  # 验证时通常针对特定模型，不需要 curriculum 逻辑

    # Curriculum learning with threshold-based transitions
    # Complexity: O3 (8 objectives) > O2 (4 objectives) > O1 (2 objectives)
    config.curriculum_schedule = [
        {"objectives": ["f1", "f8"], "complexity": "O1"},           # Stage 0: Simple (Time + Energy)
        {"objectives": ["f1", "f2", "f3", "f4"], "complexity": "O2"},  # Stage 1: Medium (+ Cost, Reliability, Quality)
        {"objectives": ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"], "complexity": "O3"}  # Stage 2: Complex (All 8 objectives)
    ]
    
    # Threshold-based transition parameters
    config.curriculum_window_size = 50     # W: sliding window size for moving average
    config.curriculum_epsilon = 0.05       # ϵ: relative improvement threshold (5%)
    config.curriculum_min_episodes = 100   # Minimum episodes per stage before transition allowed

    # 指定要验证的模型路径 (请修改为您实际的模型路径)
    target_model_path = 'xxx' 
    previous_model_path = None

    for case in [4]:  # 选择要验证的 Case，例如 Case 2 (200任务)
        if case == 2:# 计算任务数量，制造任务数量，制造任务的子任务数量，边缘设备数量，设备数量，云服务器数量
            config.environment = CEDCS_env('./instances/data_matrix_200_seed1.txt', 200, 200, 5, 200, 300, 200)
        elif case == 3:
            config.environment = CEDCS_env('./instances/data_matrix_300_seed1.txt', 300, 300, 5, 300, 300, 300)
        elif case == 4:
            config.environment = CEDCS_env('./instances/data_matrix_400_seed1.txt', 400, 400, 5, 400, 300, 400)  # seed2 3 400 seed1 300
        elif case == 5:
            config.environment = CEDCS_env('./instances/data_matrix_500_seed1.txt', 500, 500, 5, 500, 500, 500)

        # 加载指定的验证模型
        if os.path.exists(target_model_path):
            print(f"Loading model for validation: {target_model_path}")
            config.is_load = True
            config.custom_load_path = target_model_path
        else:
            print(f"Warning: Model not found at {target_model_path}. Running with random init.")
            config.is_load = False
            config.custom_load_path = None

        config.file_to_save_results_graph = './my_data_and_graph/graph' + config.environment.env_name + "_epi_rews.png"
        config.file_to_save_results_graph = './my_data_and_graph/graph' + config.environment.env_name + "_epi_rews.png"
        
        #打开记录的关键指标：Objective, Reward, Makespan, Energy, Goal
        with open('./my_data_and_graph/' + config.environment.env_name + 'logs.txt', 'w') as f:
            pass
        with open('./my_data_and_graph/losses/dnnloss.txt', 'w') as f:
            pass
        with open('./my_data_and_graph/' + config.environment.env_name + 'times.txt', 'w') as f:
            pass
        AGENTS = [TD3]  #OTDPG,DDPG_HER，TD3_HER, TD3
        # clear the logs
        for agent in AGENTS:
            with open('./my_data_and_graph/trans_' + agent.agent_name + '.txt', 'w') as f:
                pass
        trainer = Trainer(config, AGENTS)
        trainer.run_games_for_agents()
        
        # 更新上一轮模型的路径，供下一轮使用
        if use_curriculum:
            agent_name = AGENTS[0].agent_name
            previous_model_path = './my_data_and_graph/models/' + config.environment.env_name + '-' + agent_name + '-final.pt'
