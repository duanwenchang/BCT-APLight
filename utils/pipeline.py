from .generator import Generator
from .construct_sample import ConstructSample
from .updater import Updater
from . import model_test
import json
import shutil
import os
import time
from multiprocessing import Process
from models.BCT_AT_agent import critiqe_data_process
from CT.Critique import Critique 
import csv


def path_check(dic_path):
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])
    if os.path.exists(dic_path["PATH_TO_MODEL"]):
        if dic_path["PATH_TO_MODEL"] != "model/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_MODEL"])
    if "PATH_TO_PREDICT_MODEL" in dic_path:
        if os.path.exists(dic_path["PATH_TO_PREDICT_MODEL"]):
            if dic_path["PATH_TO_PREDICT_MODEL"] != "predict_model/default":
                raise FileExistsError
        else:
            os.makedirs(dic_path["PATH_TO_PREDICT_MODEL"])
    else:
        raise KeyError("PATH_TO_PREDICT_MODEL is not defined in dic_path")


def copy_conf_file(dic_path, dic_agent_conf, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"), indent=4)
    json.dump(dic_traffic_env_conf, open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)


def copy_cityflow_file(dic_path, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["TRAFFIC_FILE"]),
                os.path.join(path, dic_traffic_env_conf["TRAFFIC_FILE"]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["ROADNET_FILE"]),
                os.path.join(path, dic_traffic_env_conf["ROADNET_FILE"]))


def generator_wrapper(cnt_round, cnt_gen, dic_path, dic_agent_conf, dic_traffic_env_conf):
    generator = Generator(cnt_round=cnt_round,
                          cnt_gen=cnt_gen,
                          dic_path=dic_path,
                          dic_agent_conf=dic_agent_conf,
                          dic_traffic_env_conf=dic_traffic_env_conf,
                          )
    print("make generator")
    generator.generate(cnt_round)
    print("generator_wrapper end")
    return


def updater_wrapper(cnt_round, dic_agent_conf, dic_traffic_env_conf, dic_path):

    updater = Updater(
        cnt_round=cnt_round,
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path
    )
    updater.load_sample_for_agents()
    updater.update_network_for_agents()
    print("updater_wrapper end")
    return


class Pipeline:

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self.initialize()

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

    def run(self, multi_process=False):
        # f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "w")
        # f_time.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")
        # f_time.close()
        for cnt_round in range(self.dic_traffic_env_conf["NUM_ROUNDS"]):
            # archive_folder = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], f"archive_round_{cnt_round}")
            # os.makedirs(archive_folder, exist_ok=True)

            # original_path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
            # truncated_path = original_path[:original_path.rfind("json") + len("json")]
            # if cnt_round > 9:
            #     train_round = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
            # else:
            #     train_round = os.path.join(truncated_path, "train_round")

            #     ci_bayes95_all_intersections = []
            #     for intersection in range(12):
            #         file_name = f"total_samples_inter_{intersection}.pkl"
            #         src_file = os.path.join(train_round, file_name)
            #         if os.path.exists(src_file):
            #             rewards_series = critiqe_data_process(src_file)
            #             critique = Critique(rewards_series)
            #             critique.order_selection()
            #             ci_bayes95 = critique.forecast_future(h=10)
            #             ci_bayes95_all_intersections.append(ci_bayes95)

            #     ci_bayes95_file = os.path.join(archive_folder, f"ci_bayes95_round_{cnt_round}.csv")
            #     with open(ci_bayes95_file, 'w', newline='') as csvfile:
            #         csv_writer = csv.writer(csvfile)
            #         csv_writer.writerow(['Intersection', 'Step', 'Lower Bound', 'Upper Bound'])
            #         for idx, ci_bayes95 in enumerate(ci_bayes95_all_intersections):
            #             for step, (lower, upper) in enumerate(ci_bayes95):
            #                 csv_writer.writerow([idx, step, lower, upper])

            print("round %d starts" % cnt_round)
            round_start_time = time.time()
            process_list = []

            print("==============  generator =============")
            generator_start_time = time.time()
            if multi_process:
                print("-------------- use multi-process for generator -------------")
                for cnt_gen in range(self.dic_traffic_env_conf["NUM_GENERATORS"]):
                    p = Process(target=generator_wrapper,
                                args=(cnt_round, cnt_gen, self.dic_path,
                                      self.dic_agent_conf, self.dic_traffic_env_conf)
                                )
                    print("before")
                    p.start()
                    print("end")
                    process_list.append(p)
                print("before join")
                for i in range(len(process_list)):
                    p = process_list[i]
                    print("generator %d to join" % i)
                    p.join()
                    print("generator %d finish join" % i)
                print("end join")
            else:
                for cnt_gen in range(self.dic_traffic_env_conf["NUM_GENERATORS"]):
                    generator_wrapper(cnt_round=cnt_round,
                                      cnt_gen=cnt_gen,
                                      dic_path=self.dic_path,
                                      dic_agent_conf=self.dic_agent_conf,
                                      dic_traffic_env_conf=self.dic_traffic_env_conf)
            generator_end_time = time.time()
            generator_total_time = generator_end_time - generator_start_time

            print("==============  make samples =============")
            # make samples and determine which samples are good
            making_samples_start_time = time.time()
            train_round = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
            if not os.path.exists(train_round):
                os.makedirs(train_round)
            cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                                 dic_traffic_env_conf=self.dic_traffic_env_conf)
            cs.make_reward_for_system()
            making_samples_end_time = time.time()
            making_samples_total_time = making_samples_end_time - making_samples_start_time

            print("==============  update network =============")
            update_network_start_time = time.time()
            if self.dic_traffic_env_conf["MODEL_NAME"] in self.dic_traffic_env_conf["LIST_MODEL_NEED_TO_UPDATE"]:
                if multi_process:
                    p = Process(target=updater_wrapper,
                                args=(cnt_round,
                                      self.dic_agent_conf,
                                      self.dic_traffic_env_conf,
                                      self.dic_path))
                    p.start()
                    print("update to join")
                    p.join()
                    print("update finish join")
                else:
                    updater_wrapper(cnt_round=cnt_round,
                                    dic_agent_conf=self.dic_agent_conf,
                                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                                    dic_path=self.dic_path)

            update_network_end_time = time.time()
            update_network_total_time = update_network_end_time - update_network_start_time

            # print("==============  test evaluation =============")
            # test_evaluation_start_time = time.time()
            # model_test.test(self.dic_path["PATH_TO_MODEL"], cnt_round,
            #                 self.dic_traffic_env_conf["RUN_COUNTS"], self.dic_traffic_env_conf)

            # test_evaluation_end_time = time.time()
            # test_evaluation_total_time = test_evaluation_end_time - test_evaluation_start_time

            # print("Generator time: ", generator_total_time)
            # print("Making samples time:", making_samples_total_time)
            # print("update_network time:", update_network_total_time)
            # print("test_evaluation time:", test_evaluation_total_time)

            self.clean_up_generator_logs(cnt_round)

            # print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))
            # f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "a")
            # f_time.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(generator_total_time, making_samples_total_time,
            #                                                 update_network_total_time, test_evaluation_total_time,
            #                                                 time.time()-round_start_time))
            # f_time.close()

    def clean_up_generator_logs(self, cnt_round):
        path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                   "round_" + str(cnt_round))
        if os.path.exists(path_to_log):
            shutil.rmtree(path_to_log)
            print(f"Deleted directory: {path_to_log}")
