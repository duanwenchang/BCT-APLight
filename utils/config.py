from models.fixedtime_agent import FixedtimeAgent
from models.maxpressure_agent import MaxPressureAgent
from models.efficient_maxpressure_agent import EfficientMaxPressureAgent
from models.mplight_agent import MPLightAgent
from models.colight_agent import CoLightAgent
from models.presslight_one import PressLightAgentOne
from models.efficient_presslight_one import EfficientPressLightAgentOne
from models.efficent_presslight import EfficientPressLightAgent
from models.presslight import PressLightAgent
from models.BCT_AT_agent import BCT_AT_Agent

DIC_AGENTS = {
    "Fixedtime": FixedtimeAgent,
    "MaxPressure": MaxPressureAgent,
    "EfficientMaxPressure": EfficientMaxPressureAgent,
    "PressLight": PressLightAgent,
    "EfficientPressLight": EfficientPressLightAgent,
    "PressLightOne": PressLightAgentOne,
    "EfficientPressLightOne": EfficientPressLightAgentOne,
    "Colight": CoLightAgent,
    "MPLight": MPLightAgent,
    "BCT-AT": BCT_AT_Agent, 
}

DIC_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_ERROR": "errors/default",
    "PATH_TO_PREDICT_MODEL": "predict_model/default",
}

dic_traffic_env_conf = {

    "LIST_MODEL": ["Fixedtime",  "MaxPressure", "MPLight", "Colight", "PressLight", "PressLightOne",
                   "EfficientMaxPressure", "EfficientPressLight", "EfficientPressLightOne", "BCT-AT"],
    "LIST_MODEL_NEED_TO_UPDATE": ["MPLight", "Colight", "PressLight", "PressLightOne", "EfficientPressLight",
                                  "EfficientPressLightOne", "BCT-AT"],

    "NUM_LANE": 12,
    "PHASE_MAP": [[1, 4, 12, 13, 14, 15, 16, 17], [7, 10, 18, 19, 20, 21, 22, 23], [0, 3, 18, 19, 20, 21, 22, 23], [6, 9, 12, 13, 14, 15, 16, 17]],
    "FORGET_ROUND": 20,
    "RUN_COUNTS": 3600,
    "MODEL_NAME": None,
    "TOP_K_ADJACENCY": 5,
    "OBS_LENGTH": 167,

    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 1,

    "MIN_ACTION_TIME": 30,
    "MEASURE_TIME": 30,
 
    "BINARY_PHASE_EXPANSION": True,

    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 8,
    "NUM_LANES": [3, 3, 3, 3],

    "INTERVAL": 1,

    "LIST_STATE_FEATURE": [
        "cur_phase",
        "time_this_phase",
        "lane_num_vehicle_upstream",
        "lane_num_vehicle_next_upstream",
        "lane_num_waiting_vehicle_in",
        "lane_num_waiting_vehicle_out",
        "lane_enter_running_part",
        "lane_exit_running_part",
        "downstream_pressure",
        "pressure",
        "adjacency_matrix"
        
    ],
    "DIC_REWARD_INFO": {
        "queue_length": 0,
        "pressure": 0,
    },
    "PHASE": {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0],
        5: [1, 1, 0, 0, 0, 0, 0, 0],
        6: [0, 0, 1, 1, 0, 0, 0, 0],
        7: [0, 0, 0, 0, 0, 0, 1, 1],
        8: [0, 0, 0, 0, 1, 1, 0, 0]
        },
    "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],
    "PHASE_LIST": ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL', 'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT'],

    "STATE_DIM": None,  
    "ACTION_DIM": 8,  
}

DIC_BASE_AGENT_CONF = {
    "D_DENSE": 20,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 10,
    "BATCH_SIZE": 20,  
    "EPOCHS": 100,
    "SAMPLE_SIZE": 3000,
    "MAX_MEMORY_LEN": 6000,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "NORMAL_FACTOR": 20,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
}

DIC_FIXEDTIME_AGENT_CONF = {
    "FIXED_TIME": [15, 15, 15, 15]
}

DIC_MAXPRESSURE_AGENT_CONF = {
    "FIXED_TIME": [15, 15, 15, 15]
}