import pickle
import numpy as np
import os
import csv
import ast

import pandas as pd
from .agent import Agent
import random
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Lambda, Layer, Reshape
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from CT.Critique import Critique
from CT.Tune import Tune


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=3, num_heads=1)
        
        self.PHASE = {
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0],
            5: [1, 1, 0, 0, 0, 0, 0, 0],
            6: [0, 0, 1, 1, 0, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 1, 1],
            8: [0, 0, 0, 0, 1, 1, 0, 0]
        }

    def get_features(self, state):
        device = next(self.parameters()).device
        cur_phase = state['cur_phase']
        if len(cur_phase) == 8:
            feat1 = torch.tensor(cur_phase, dtype=torch.float32).to(device)
        else:
            cur_phase_binary = self.PHASE[cur_phase[0]]
            feat1 = torch.tensor(cur_phase_binary, dtype=torch.float32).to(device)
        feat2_pre = state['lane_enter_running_part']
        feat2 = torch.tensor(feat2_pre, dtype=torch.float32).to(device)
        query_group = [
            [state['lane_num_vehicle_upstream'][i:i+3] for i in range(0, len(state['lane_num_vehicle_upstream']), 3)],
            [state['lane_num_waiting_vehicle_out'][i:i+3] for i in range(0, len(state['lane_num_waiting_vehicle_out']), 3)],
            [state['lane_exit_running_part'][i:i+3] for i in range(0, len(state['lane_exit_running_part']), 3)]
        ]
        
        key_group_pre = [
            [state['lane_num_vehicle_next_upstream'][i:i+3] for i in range(0, len(state['lane_num_vehicle_next_upstream']), 3)],
            [state['lane_num_waiting_vehicle_in'][i:i+3] for i in range(0, len(state['lane_num_waiting_vehicle_in']), 3)],
            [state['lane_enter_running_part'][i:i+3] for i in range(0, len(state['lane_enter_running_part']), 3)]
        ]
        key_group = []
        for i in range(3):
            key_group.append([
                [key_group_pre[i][2][0], key_group_pre[i][0][1], key_group_pre[i][3][2]],
                [key_group_pre[i][3][0], key_group_pre[i][1][1], key_group_pre[i][2][2]],
                [key_group_pre[i][1][0], key_group_pre[i][2][1], key_group_pre[i][0][2]],
                [key_group_pre[i][0][0], key_group_pre[i][3][1], key_group_pre[i][1][2]]
            ])
        all_directions_weights = []
        for j in range(4):
            key_group_list = [key_group[0][j], key_group[1][j], key_group[2][j]]
            query_group_list = [query_group[0][j], query_group[1][j], query_group[2][j]]

            key_tensor = torch.tensor([list(i) for i in zip(*key_group_list)], dtype=torch.float32).unsqueeze(1).to(device)
            query_list = [list(i) for i in zip(*query_group_list)]
            query_tensor = torch.tensor(query_list, dtype=torch.float32).unsqueeze(1).to(device)

            _, attn_output_weights = self.multihead_attn(query_tensor, key_tensor, key_tensor)
            attn_output_weights_list = attn_output_weights.squeeze().tolist()
            weight = list(map(list, zip(*attn_output_weights_list)))
            all_directions_weights.extend(weight)

        attention_pressure_pre = query_group[1]
        attention_queue_length = []
        for i in range(0, len(all_directions_weights), 3):
            pressure_sublist = attention_pressure_pre[i // 3]
            multiplied_values = []
            for j in range(3):
                multiplied_sum = sum(
                    all_directions_weights[i + j][k] * pressure_sublist[k]
                    for k in range(len(pressure_sublist))
                )
                multiplied_values.append(multiplied_sum)
            attention_queue_length.extend(multiplied_values)

        attention_pressure = _get_traffic_movement_pressure_efficient(
            state['lane_num_waiting_vehicle_in'], attention_queue_length)
        feat3 = torch.tensor(attention_pressure, dtype=torch.float32).to(device)
        feat4 = torch.tensor(state['lane_exit_running_part'], dtype=torch.float32).to(device)
        feat5 = torch.tensor(state['lane_num_waiting_vehicle_in'], dtype=torch.float32).to(device)
        feat6 = torch.tensor(state['lane_num_waiting_vehicle_out'], dtype=torch.float32).to(device)

        return feat1, feat2, feat3, feat4, feat5, feat6
    

class BCT_AT_Agent(Agent):
    def __init__(self, dic_agent_conf=None, dic_traffic_env_conf=None, dic_path=None, cnt_round=None,
                 intersection_id="0"):
        super(BCT_AT_Agent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)
        self.CNN_layers = dic_agent_conf['CNN_layers']
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)

        self.num_actions = 8
        self.memory = build_memory()

        if cnt_round == 0:
            self.q_network = self.build_network()
            original_path = self.dic_path["PATH_TO_MODEL"]
            truncated_path = original_path[:original_path.rfind("json") + len("json")]
            if os.listdir(truncated_path):
                self.q_network.load_weights(
                    os.path.join(truncated_path, "round_0_inter_{0}.h5".format(intersection_id)),
                    by_name=True)
            self.q_network_bar = self.build_network_from_copy(self.q_network)
            
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))
                if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                    if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf[
                                "UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                    else:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                else:
                    self.load_network_bar("round_{0}_inter_{1}".format(
                        max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
            except:
                print("fail to load network, current round: {0}".format(cnt_round))

        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

        self.prediction_network = Predict_MLP()
        original_path = self.dic_path["PATH_TO_MODEL"]
        truncated_path = original_path[:original_path.rfind("json") + len("json")]
        if os.listdir(truncated_path):
            weight_file_path = os.path.join(truncated_path, f"round_0_inter_0_predict.pth")
            if os.path.exists(weight_file_path):
                checkpoint = torch.load(weight_file_path)
                self.prediction_network.load_state_dict(checkpoint)
            else:
                raise FileNotFoundError(f"Weight file not found: {weight_file_path}")

    @staticmethod
    def MLP(ins, layers=None):
        if layers is None:
            layers = [128, 128]
        for layer_index, layer_size in enumerate(layers):
            if layer_index == 0:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal',
                          name='Dense_embed_%d' % layer_index)(ins)
            else:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal',
                          name='Dense_embed_%d' % layer_index)(h)
        return h

    def MultiHeadsAttModel(self, in_feats, in_nei, d_in=128, h_dim=16, dout=128, head=8, suffix=-1):
        agent_repr = Reshape((self.num_agents, 1, d_in))(in_feats)
        neighbor_repr = RepeatVector3D(self.num_agents)(in_feats)
        neighbor_repr = Lambda(lambda x: tf.matmul(x[0], x[1]))([in_nei, neighbor_repr])
        agent_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                name='agent_repr_%d' % suffix)(agent_repr)
        agent_repr_head = Reshape((self.num_agents, 1, h_dim, head))(agent_repr_head)
        agent_repr_head = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 4, 2, 3)))(agent_repr_head)
        neighbor_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                   name='neighbor_repr_%d' % suffix)(neighbor_repr)
        neighbor_repr_head = Reshape((self.num_agents, self.num_neighbors, h_dim, head))(neighbor_repr_head)
        neighbor_repr_head = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 4, 2, 3)))(neighbor_repr_head)
        att = Lambda(lambda x: K.softmax(tf.matmul(x[0], x[1], transpose_b=True)))([agent_repr_head,
                                                                                    neighbor_repr_head])
        att_record = Reshape((self.num_agents, head, self.num_neighbors))(att)
        neighbor_hidden_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                          name='neighbor_hidden_repr_%d' % suffix)(neighbor_repr)
        neighbor_hidden_repr_head = Reshape((self.num_agents, self.num_neighbors, h_dim, head))(
            neighbor_hidden_repr_head)
        neighbor_hidden_repr_head = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 4, 2, 3)))(
            neighbor_hidden_repr_head)
        out = Lambda(lambda x: K.mean(tf.matmul(x[0], x[1]), axis=2))([att, neighbor_hidden_repr_head])
        out = Reshape((self.num_agents, h_dim))(out)
        out = Dense(dout, activation="relu", kernel_initializer='random_normal', name='MLP_after_relation_%d' % suffix)(
            out)
        return out, att_record

    def adjacency_index2matrix(self, adjacency_index):
        adjacency_index_new = np.sort(adjacency_index, axis=-1)
        lab = to_categorical(adjacency_index_new, num_classes=self.num_agents)
        return lab

    def convert_state_to_input(self, s):
        feats0 = []
        adj = []
        multi_head_attention = MultiHeadAttention()

        for i in range(self.num_agents):
            adj.append(s[i]["adjacency_matrix"])
            feat1, feat2, feat3, _, _, _ = multi_head_attention.get_features(s[i])
            tmp = np.concatenate([feat1.numpy(), feat2.numpy(), feat3.numpy()])
            feats0.append(tmp)

        feats = np.array([feats0])
        adj = self.adjacency_index2matrix(np.array([adj]))
        return [feats, adj]

    def choose_action(self, step_num, states, cnt_round):
        xs = self.convert_state_to_input(states)
        q_values = self.q_network(xs)
        multi_head_attention = MultiHeadAttention()

        combined_features = []
        for i, state in enumerate(states):
            feat1, feat2, _, feat4, feat5, feat6 = multi_head_attention.get_features(state)

            q_value_for_intersection = q_values[0, i, :].numpy() 
            q_value_tensor = torch.tensor(q_value_for_intersection, dtype=torch.float32)

            combined_feature = torch.cat((feat1, feat2, feat4, feat5, feat6, q_value_tensor), dim=0)
            combined_features.append(combined_feature)

        combined_features_tensor = torch.stack(combined_features)
        predicted_rewards = self.prediction_network(combined_features_tensor)
        predicted_rewards_list = predicted_rewards.squeeze().tolist()
        # print(f"预测奖励: {predicted_rewards_list}")
        
        archive_folder = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], f"archive_round_{cnt_round}")
        ci_bayes95_file = os.path.join(archive_folder, f"ci_bayes95_round_{cnt_round}.csv")

        original_path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        truncated_path = original_path[:original_path.rfind("json") + len("json")]
        if cnt_round > 10:
            train_round_path = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
        else:
            train_round_path = os.path.join(truncated_path, "train_round")

        if random.random() <= 0.2:
            actions = np.random.randint(self.num_actions, size=len(q_values[0]))
            for idx in range(12):
                file_name = f"total_samples_inter_{idx}.pkl"
                pkl_file = os.path.join(train_round_path, file_name)
                q_his = self.extract_historical_q_values(pkl_file)
                q_cur = q_values.numpy().tolist()[0][idx]
                tune = Tune(q_his, q_cur)
                optimal_decisions = tune.tune_decision()
                actions[idx] = optimal_decisions
        else:
            actions = np.argmax(q_values[0], axis=1)
            if os.path.exists(ci_bayes95_file):
                with open(ci_bayes95_file, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    next(csv_reader)
                    ci_bayes95_data = {int(row[0]): [(int(row[1]), float(row[2]), float(row[3]))] for row in csv_reader}

                for idx in range(12):
                    if step_num > 33:
                        interval_index = (step_num - 10) // 23
                        for step, lower_bound, upper_bound in ci_bayes95_data[idx]:
                            if step == interval_index:
                                predicted_reward = predicted_rewards_list[idx]
                                if lower_bound <= predicted_reward <= upper_bound:
                                    print(f"Predicted reward {predicted_reward} is within the interval for intersection {idx}.")
                                else:
                                    file_name = f"total_samples_inter_{idx}.pkl"
                                    pkl_file = os.path.join(train_round_path, file_name)
                                    q_his = self.extract_historical_q_values(pkl_file)
                                    q_cur = q_values.numpy().tolist()[0][idx]
                                    tune = Tune(q_his, q_cur)
                                    optimal_decisions = tune.tune_decision()
                                    actions[idx] = optimal_decisions

        return actions, q_values
    

    def choose_action_with_value(self, count, states):
        xs = self.convert_state_to_input(states)
        q_values = self.q_network(xs)
        if random.random() <= self.dic_agent_conf["EPSILON"]:
            action = np.random.randint(self.num_actions, size=len(q_values[0]))
        else:
            action = np.argmax(q_values[0], axis=1)

        norm_values = torch.softmax(torch.tensor(np.array(q_values[0])) / 0.05, dim=1)
        norm_values = (norm_values - torch.min(norm_values, dim=1)[0].unsqueeze(1)) / (torch.max(norm_values, dim=1)[0].unsqueeze(1) - torch.min(norm_values, dim=1)[0].unsqueeze(1))
        norm_values = norm_values.numpy()

        return action, norm_values

    @staticmethod
    def _concat_list(ls):
        tmp = []
        for i in range(len(ls)):
            tmp += ls[i]
        return [tmp]

    def prepare_Xs_Y(self, memory):
        slice_size = len(memory[0])
        _adjs = []
        _state = [[] for _ in range(self.num_agents)]
        _next_state = [[] for _ in range(self.num_agents)]
        _action = [[] for _ in range(self.num_agents)]
        _reward = [[] for _ in range(self.num_agents)]

        multi_head_attention = MultiHeadAttention()

        for i in range(slice_size):
            _adj = []
            for j in range(self.num_agents):
                state, action, next_state, reward, _, _, _, _ = memory[j][i]
                _action[j].append(action)
                _reward[j].append(reward)
                _adj.append(state["adjacency_matrix"])

                feat1, feat2, feat3, _, _, _ = multi_head_attention.get_features(state)
                state_features = torch.cat([feat1, feat2, feat3]).tolist()
                _state[j].append([state_features])
                next_feat1, next_feat2, next_feat3,_,_,_ = multi_head_attention.get_features(next_state)
                next_state_features = torch.cat([next_feat1, next_feat2, next_feat3]).tolist()
                _next_state[j].append([next_state_features])

            _adjs.append(_adj)

        _adjs2 = self.adjacency_index2matrix(np.array(_adjs))
        _state2 = np.concatenate([np.array(ss) for ss in _state], axis=1)
        _next_state2 = np.concatenate([np.array(ss) for ss in _next_state], axis=1)
        target = self.q_network([_state2, _adjs2])
        next_state_qvalues = self.q_network_bar([_next_state2, _adjs2])
        final_target = np.copy(target)
        for i in range(slice_size):
            for j in range(self.num_agents):
                final_target[i, j, _action[j][i]] = _reward[j][i] / self.dic_agent_conf["NORMAL_FACTOR"] + \
                                                    self.dic_agent_conf["GAMMA"] * np.max(next_state_qvalues[i, j])

        self.Xs = [_state2, _adjs2]
        self.Y = final_target

    def build_network(self, MLP_layers=[32, 32]):
        CNN_layers = self.CNN_layers
        CNN_heads = [5] * len(CNN_layers)
        In = list()
        In.append(Input(shape=(self.num_agents, 32), name="feature"))
        In.append(Input(shape=(self.num_agents, self.num_neighbors, self.num_agents), name="adjacency_matrix"))

        feature = self.MLP(In[0], MLP_layers)
        print("CNN_heads:", CNN_heads)
        for CNN_layer_index, CNN_layer_size in enumerate(CNN_layers):
            print("CNN_heads[CNN_layer_index]:", CNN_heads[CNN_layer_index])
            if CNN_layer_index == 0:
                h, _ = self.MultiHeadsAttModel(
                    feature,
                    In[1],
                    d_in=MLP_layers[-1],
                    h_dim=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    head=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                )
            else:
                h, _ = self.MultiHeadsAttModel(
                    h,
                    In[1],
                    d_in=MLP_layers[-1],
                    h_dim=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    head=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                )
        out = Dense(self.num_actions, kernel_initializer='random_normal', name='action_layer')(h)
        model = Model(inputs=In, outputs=out)

        model.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                      loss=self.dic_agent_conf["LOSS_FUNCTION"])
        model.summary()
        return model

    def train_network(self):
        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')
        self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs, shuffle=False,
                           verbose=2, validation_split=0.3, callbacks=[early_stopping])
    
    def extract_historical_q_values(self, pkl_file_path):
        num_partitions = 10
        data_frames = []
    
        with open(pkl_file_path, "rb") as f:
            try:
                while True:
                    data = pickle.load(f)
                    data_frames.append(pd.DataFrame(data))
            except EOFError:
                pass

        recent_data_frames = data_frames[-num_partitions:]
        combined_data = pd.concat(recent_data_frames, ignore_index=True)

        values_to_remove = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
        combined_data = combined_data[~combined_data.iloc[:, 6].isin(values_to_remove)]

        q_his_array = combined_data.iloc[:, 5].to_numpy()
        split_q = np.array_split(q_his_array, len(q_his_array) // 11)

        selected_q = split_q[::10]
        selected_q_combined = np.concatenate(selected_q).tolist()
        transposed_list = [list(i) for i in zip(*selected_q_combined)]
        
        return transposed_list

    def build_network_from_copy(self, network_copy):
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={"RepeatVector3D": RepeatVector3D})
        network.set_weights(network_weights)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])

        return network

    def build_network_from_copy_only_weight(self, network, network_copy):
        network_weights = network_copy.get_weights()
        network.set_weights(network_weights)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        return network

    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        self.q_network = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={'RepeatVector3D': RepeatVector3D})
        print("succeed in loading model %s" % file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={'RepeatVector3D': RepeatVector3D})
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def save_network_bar(self, file_name):
        self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def save_predict_network(self, file_name):
        path = self.dic_path["PATH_TO_PREDICT_MODEL"]
        os.makedirs(path, exist_ok=True)

        model_path = os.path.join(path, f"{file_name}.pth")
        torch.save(self.prediction_network.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load_predict_network(self, file_name, device):
        path = self.dic_path["PATH_TO_PREDICT_MODEL"]
        model_path = os.path.join(path, f"{file_name}.pth")
        self.prediction_network.load_state_dict(torch.load(model_path))
        self.prediction_network.to(device)
        print(f"Succeeded in loading prediction model {file_name}")


class RepeatVector3D(Layer):
    def __init__(self, times, **kwargs):
        super(RepeatVector3D, self).__init__(**kwargs)
        self.times = times

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.times, input_shape[1], input_shape[2]

    def call(self, inputs):
        return K.tile(K.expand_dims(inputs, 1), [1, self.times, 1, 1])

    def get_config(self):
        config = {'times': self.times}
        base_config = super(RepeatVector3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def critiqe_data_process(pkl_file_path):
    num_partitions = 10
    data_frames = []
    
    with open(pkl_file_path, "rb") as f:
        try:
            while True:
                data = pickle.load(f)
                data_frames.append(pd.DataFrame(data))
        except EOFError:
            pass

    recent_data_frames = data_frames[-num_partitions:]
    combined_data = pd.concat(recent_data_frames, ignore_index=True)

    values_to_remove = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
    combined_data = combined_data[~combined_data.iloc[:, 6].isin(values_to_remove)]

    rewards = combined_data.iloc[:, 4]
    rewards_array = rewards.to_numpy()
    split_rewards = np.array_split(rewards_array, len(rewards_array) // 11)
    rewards_mean = [np.mean(group) for group in split_rewards]

    start_date = '2024-01-01'
    freq = '15T'
    timestamps = pd.date_range(start=start_date, periods=len(rewards_mean), freq=freq)
    reward_series = pd.Series(rewards_mean, index=timestamps)

    return reward_series

def _get_traffic_movement_pressure_efficient(enterings, exitings):
        """
            Created by LiangZhang
            Calculate pressure with entering and exiting vehicles
            only for 3 x 3 lanes intersection
        """
        list_approachs = ["W", "E", "N", "S"]
        index_maps = {
            "W": [0, 1, 2],
            "E": [3, 4, 5],
            "N": [6, 7, 8],
            "S": [9, 10, 11]
        }
        # vehicles in exiting road
        outs_maps = {}
        for approach in list_approachs:
            outs_maps[approach] = sum([exitings[i] for i in index_maps[approach]])
        turn_maps = ["S", "W", "N", "N", "E", "S", "W", "N", "E", "E", "S", "W"]
        t_m_p = [enterings[j] - outs_maps[turn_maps[j]] / 3 for j in range(12)]
        return t_m_p

def build_memory():
    return []

class Predict_MLP(nn.Module):
    def __init__(self):
        super(Predict_MLP, self).__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, flattened_input_q):
        x = self.relu(self.fc1(flattened_input_q))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def train_predict_network(pkl_folder):
    prediction_network = Predict_MLP()
    optimizer = torch.optim.Adam(prediction_network.parameters(), lr=0.001)
    prediction_network.train()

    sampled_samples = sample_from_latest_partitions(pkl_folder, 64)
    combined_list, reward_list = extract_features_predict_network(sampled_samples)
    
    reward_tensor = torch.tensor(reward_list, dtype=torch.float32)
    combined_tensor = torch.tensor(combined_list,dtype=torch.float32)

    optimizer.zero_grad()
    predicted_rewards = prediction_network(combined_tensor)
    loss = F.mse_loss(predicted_rewards, reward_tensor)
    loss.backward()
    optimizer.step()


def sample_from_latest_partitions(pkl_folder, sample_size):
    all_samples = []
    pkl_files = [os.path.join(pkl_folder, f"total_samples_inter_{i}.pkl") for i in range(12)]

    for file in pkl_files:
        try:
            with open(file, "rb") as f:
                last_partition = None
                while True:
                    try:
                        last_partition = pickle.load(f)
                    except EOFError:
                        break

                if last_partition is not None:
                    all_samples.extend(last_partition) 
        except Exception as e:
            print(f"Error processing {file}: {e}")

    sampled_data = random.sample(all_samples, sample_size)
    return sampled_data

def extract_features_predict_network(states):
    combined_list = []
    reward_list = []
    for sample in states:
        state = sample[0]
        reward = float(sample[4])
        q_values = sample[5]

        feat1 = state['cur_phase']
        feat2 = state['lane_enter_running_part']
        feat3 = state['lane_exit_running_part']
        feat4 = state['lane_num_waiting_vehicle_in']
        feat5 = state['lane_num_waiting_vehicle_out']

        combined_features = feat1 + feat2 + feat3 + feat4 + feat5 + q_values
        reward_list.append(reward)
    return combined_list, reward_list







