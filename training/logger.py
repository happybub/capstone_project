import matplotlib.pyplot as plt
import torch
import math

class Logger:
    def __init__(self, path, name):
        self.name = name
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

    def get_values(self, key, cal_mean=False):
        ret = [log.get(key, float('nan')) for log in self.logs]
        if cal_mean:
            return sum([x for x in ret if not math.isnan(x)]) / len(ret)
        return ret

    def str_line(self, idx=-1):
        l = self.logs[idx]
        s = ', '.join([f'{key}: {l[key]}' for key in l.keys()])
        return s

    @classmethod
    def line_graph(cls, logger, keys, save_path=None):
        """Draw multiple keys from a single Logger instance on one plot."""
        plt.figure()
        for key in keys:
            plt.plot(logger.get_values(key), label=key)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Values')
        plt.title('Multiple Metrics for ' + logger.name)
        plt.show()
        if save_path:
            plt.savefig(save_path)

    @classmethod
    def multi_logger_graph(cls, loggers, keys):
        """Draw line graphs for each key from multiple Logger instances."""
        for key in keys:
            plt.figure()
            plt.title(key)
            for logger in loggers:
                values = logger.get_values(key)
                if any(not isinstance(v, float) or not math.isnan(v) for v in values):
                    plt.plot(values, label=logger.name)
            plt.ylabel(key)
            plt.xlabel('Epoch')
            plt.legend()
            plt.show()

    def save_to_file(self, mode='a'):
        with open(self.path, mode) as f:
            for i in range(self.saved_lines, len(self.logs)):
                f.write(f'{self.str_line(i)}\n')
            self.saved_lines = len(self.logs)


    def load_from_file(self):
        if len(self.logs) > 0:
            return
        with open(self.path, 'r') as file:
            for line in file:
                log_entry = {}
                items = line.split(', ')
                for item in items:
                    key, value = item.split(': ')
                    try:
                        log_entry[key] = float(value)
                    except ValueError:
                        log_entry[key] = value.strip()
                self.logs.append(log_entry)

if __name__ == '__main__':
    name = 'y vit; f dense; with occlusion'
    y_vit = Logger(f'../logs/{name}/val_logs.log', 'y as vit with occlusion attack')
    y_vit.load_from_file()

    Logger.line_graph(y_vit, ["Secret", "Image", "Acc"])
    Logger.multi_logger_graph([y_vit], ["Secret", "Image", "Acc"])