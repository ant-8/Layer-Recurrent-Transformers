from train import run
import argparse
import json
import os
from types import SimpleNamespace

class SafeSimpleNamespace(SimpleNamespace):
    def __getattr__(self, name):
        return None

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    args = parser.parse_args()
    return args

def load_experiment_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

args = parse_args()
config = load_experiment_config(args.config)
results = []
experiment_name = os.path.splitext(os.path.basename(args.config))[0]

start = args.start or config['shared_args'].get("resume_index", 0)
end = args.end or len(config['runs'])

for run_config in config['runs'][start:end]:
    if run_config.get("skip", False): continue
    run_config['project_name'] = experiment_name
    run_config['device'] = args.device
    for key in config['shared_args']:
        value = config['shared_args'][key]
        if key not in run_config:
            run_config[key] = value
    #print(run_config)
    run_config = SafeSimpleNamespace(**run_config)
    run_results = run(run_config)
    results.append(run_results)

output_dir = f"./experiment_results/"
os.makedirs(output_dir, exist_ok=True)
filename = f"./experiment_results/{experiment_name}.json"
with open(filename, "w") as f:
    json.dump(results, f, indent=2)