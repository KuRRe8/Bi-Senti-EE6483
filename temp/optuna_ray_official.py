from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
import ray
import optuna
import os

def objective(config):  #
    target_val = 100
    running_total = 0
    print('here starting a trail')
    print('*'*50)
    while True:
        running_total += config['num1']
        train.report({"distance_to_target": target_val- running_total})  # Report to Tune


#search_space = {"lr": tune.loguniform(1e-4, 1e-2), "momentum": tune.uniform(0.1, 0.9)}
search_space = {"num1": tune.uniform(0, 1)}

algo = OptunaSearch()  #

tuner = tune.Tuner(  #
    objective,
    tune_config=tune.TuneConfig(
        metric="distance_to_target", # must be reported by the objective function
        mode="min", #  "min" or "max" to minimize or maximize the metric
        search_alg=algo, # Defaults to random search
        num_samples=10,
    ),
    run_config=train.RunConfig(
        name="finally find you",
        storage_path=os.path.join(os.getcwd(), "temp", "optuna_ray_official_storage"),
        stop={"training_iteration": 100},
    ),
    param_space=search_space,
)
results = tuner.fit()
print("Best config is:", results.get_best_result().config)