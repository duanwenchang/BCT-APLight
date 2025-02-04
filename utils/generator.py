from .config import DIC_AGENTS
from .cityflow_env import CityFlowEnv
import time
import os
import copy
import pickle
from itertools import chain
import logging
import numpy as np


class Generator:
    def __init__(self, cnt_round, cnt_gen, dic_path, dic_agent_conf, dic_traffic_env_conf):

        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.agents = [None]*dic_traffic_env_conf['NUM_AGENTS']
        self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                        "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)
            start_time = time.time()
            for i in range(dic_traffic_env_conf['NUM_AGENTS']):
                agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=self.dic_agent_conf,
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    dic_path=self.dic_path,
                    cnt_round=self.cnt_round,
                    intersection_id=str(i)
                )
                self.agents[i] = agent
            print("Create intersection agent time: ", time.time()-start_time)

        self.results_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.env = CityFlowEnv(
            path_to_log=self.path_to_log,
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf
        )

    def generate(self, cnt_round):
        reset_env_start_time = time.time()
        done = False
        state_list = self.env.reset()
        step_num = 0
        reset_env_time = time.time() - reset_env_start_time
        running_start_time = time.time()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []
        q_values_history = [[] for _ in range(12)]

        while not done and step_num < int(self.dic_traffic_env_conf["RUN_COUNTS"] /
                                          self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []
            step_start_time = time.time()
            for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):
                if self.dic_traffic_env_conf["MODEL_NAME"] in ["MPLight", "Colight", "PressLightOne", "EfficientPressLightOne", "BCT-AT"]:
                    one_state_list = state_list
                    actions, q_values = self.agents[i].choose_action(step_num, one_state_list, cnt_round)
                    q = q_values.numpy().tolist()[0]
                    action_list = actions
            for i, q_sublist in enumerate(q):
                q_values_history[i].append(q_sublist)

            next_state_list, reward, done, _ = self.env.step(action_list)

            print("time: {0}, running_time: {1}".format(self.env.get_current_time() -
                                                        self.dic_traffic_env_conf["MIN_ACTION_TIME"],
                                                        time.time()-step_start_time))

            total_reward += sum(reward)
            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter))

            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

            state_list = next_state_list
            step_num += 1


        print("start logging.......................")
        self.env.bulk_log_multi_process()

        for inter_ind in range(12):
            path_to_log_file = os.path.join(self.path_to_log, f"inter_{inter_ind}.pkl")
            with open(path_to_log_file, 'rb') as f:
                log_data = pickle.load(f)
            for time_step, time_step_data in enumerate(log_data):
                try:
                    index = time_step // self.dic_traffic_env_conf["MIN_ACTION_TIME"]
                    time_step_data['q_values'] = q_values_history[inter_ind][index]
                except IndexError as e:
                    print(f"IndexError at agent {inter_ind}, time_step {time_step}: {str(e)}")
                    print(f"q_values_history[{inter_ind}] 长度 = {len(q_values_history[inter_ind])}")
                    raise
            with open(path_to_log_file, 'wb') as f:
                pickle.dump(log_data, f)

        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time) and not np.isnan(leave_time):
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        results = {
            "training_reward": total_reward,
            "training_avg_queue_len": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "training_queuing_vehicle_num": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "training_avg_waiting_time": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "training_avg_travel_time": total_travel_time}
        
        results_file_path = os.path.join(self.results_dir, "results.pkl")
        if os.path.exists(results_file_path):
            with open(results_file_path, 'rb') as f:
                existing_results = pickle.load(f)
        else:
            existing_results = []
            
        existing_results.append(results)
        
        with open(results_file_path, 'wb') as f:
            pickle.dump(existing_results, f)
        logging.info(f"Results for round {cnt_round} saved successfully.")
        
        self.env.end_cityflow()

        

        

