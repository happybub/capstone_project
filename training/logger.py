import matplotlib.pyplot as plt
import torch
import os

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
RESET = "\033[0m"



class Logger:
    def __init__(self, path):
        self.logs = []
        self.current_log = {}
        self.path = path
        self.saved_lines = 0

    def log(self, name, value):
        self.current_log[name] = value
        return self

    def save(self):
        self.logs.append(self.current_log)
        self.current_log = {}

    def format_value(self, idx, key, compare=False):
        cur_value = self.logs[idx][key]
        pre_value = None
        if len(self.logs) > 1:
            pre_value = self.logs[idx - 1][key]

        # comparable
        if type(cur_value) == type(pre_value) and isinstance(cur_value, float) and compare:
            if cur_value < pre_value:
                # format to .4f
                return f'{GREEN}{cur_value:.4f}{RESET}'
            elif cur_value > pre_value:
                return f'{RED}{cur_value:.4f}{RESET}'
            else:
                return f'{cur_value:.4f}'

        if isinstance(cur_value, float):
            return f'{cur_value:.4f}'

        return str(cur_value)

    def format_log(self, idx=-1, compare=False):
        # sort the keys
        return ', '.join([f'{key}: {self.format_value(idx, key, compare)}' for key in self.logs[idx].keys()])

    def get_values(self, key):
        return [log[key] for log in self.logs]

    def get_values_mean(self, key):
        return torch.tensor(self.get_values(key)).mean().item()

    def line_graph(self, key, save_path=None):
        # pop up the line graph, if save_path is not None, save the graph to the path
        if isinstance(key, str):
            plt.plot(self.get_values(key))
            plt.ylabel(key)
            plt.show()
            if save_path is not None:
                plt.savefig(save_path)

        # if the parameter key is a list of keys, draw them in a same graph
        elif isinstance(key, list):
            for k in key:
                plt.plot(self.get_values(k), label=k)
            plt.legend()
            plt.show()
            if save_path is not None:
                plt.savefig(save_path)

    def save_to_file(self, mode='a'):
        with open(self.path, mode) as f:
            for i in range(self.saved_lines, len(self.logs)):
                f.write(f'{self.format_log(i)}\n')
            self.saved_lines = len(self.logs)

    def load_from_file(self):
        with open(self.path, 'r') as file:
            for line in file:
                log_entry = {}
                items = line.split(', ')
                for item in items:
                    key, value = item.split(': ')
                    # Assuming values that can be converted to float should be
                    try:
                        log_entry[key] = float(value)
                    except ValueError:
                        log_entry[key] = value.strip()
                self.logs.append(log_entry)


if __name__ == '__main__':
    log_plt_path = "../log_plots/"
    model_name = '12_avgpolgan' + '/'
    start = 40

    #types = ['Image', 'Secret', 'Total', 'Acc']
    os.makedirs("log_plt_path", exist_ok=True)

    log_path = '../logs/' + model_name

    os.makedirs(log_plt_path + model_name, exist_ok=True)

    logger1 = Logger(log_path + '/val_logs.log')
    logger1.load_from_file()
    logger1.line_graph(['Real Acc', 'Fake Acc'])


    # logger2 = Logger(log_path + '/train_logs.log')
    # #logger3 = Logger('../logs/vit/train_logs.log')
    # logger1.load_from_file()
    # # logger2.load_from_file()
    # #logger3.load_from_file()
    #
    # type_dic = logger2.logs[0]
    # keys = list(type_dic.keys())
    #
    # keys_to_remove = ['epoch', 'mode']
    # for key in keys_to_remove:
    #     if key in type_dic:
    #         del type_dic[key]
    #
    # for t in type_dic:
    #     valid = logger2.get_values(t)
    #     train = logger1.get_values(t)
    #
    #     # plot
    #     plt.plot(valid, label='valid')
    #     plt.plot(train, label='train')
    #
    #     plt.ylabel(t)
    #     plt.legend()
    #     plt.savefig(log_plt_path + model_name + t + '.png', format='png', dpi=300)
    #     plt.clf()
    #
    #     _valid = logger2.get_values(t)[start:]
    #     _train = logger1.get_values(t)[start:]
    #
    #     # plot
    #     plt.plot(_valid, label='valid')
    #     plt.plot(_train, label='train')
    #
    #     plt.ylabel(t)
    #     plt.legend()
    #     plt.savefig(log_plt_path + model_name + '_' + t + '.png', format='png', dpi=300)
    #     plt.clf()
