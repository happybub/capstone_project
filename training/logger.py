import matplotlib.pyplot as plt
import torch

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
    logger = Logger('../logs/241110_161636/val_logs.log')
    logger.load_from_file()
    logger.line_graph(["Secret", "Image"])
    logger.line_graph(["Acc"])

