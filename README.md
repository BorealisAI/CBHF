# CBHF

Contains code for contextual bandits with human feedback

### Installation

To install the dependencies of the code in a virtual environment run the setup script.

    bash setup.sh

### Usage

Activate the virtual environment installed using the following command:

    source python-vms/cbhf/bin/activate

To run `ppo` using `Bibtex` dataset with `action_recommendation` with entropy threshold of `5` and expert accuracy of `0.8` use the following command

    cd src
    python main.py --env_name bibtex --algorithm ppo --human_feedback action_recommendation --feedback_interval entropy --entropy_threshold 2.5 --expert_accuracy 0.8

The program accepts the following command line arguments:

| Option                | Description                                                                                                                                                                                                 |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--env_name`          | The name of the dataset. Possible choices are `bibtex`, `mediamill`, `delicious`                                                                                                                            |
| `--algorithm`         | The algorithm to use. Possible choices are `ppo`, `ppo-lstm`, `reinforce`, `actor-critic`, `ee-net`,`tamer`, `bootstrapped-ts`, `linearucb`                                                                 |
| `--seed`              | Sets gym pytorch and numpy seeds                                                                                                                                                                            |
| `--timesteps`         | The timesteps for the algorithm                                                                                                                                                                             |
| `--eval_freq`         | The frequency at which seperate montecarlo evaluations are run                                                                                                                                              |
| `--t_horizon`         | The maximum number of timesteps to run the environment for                                                                                                                                                  |
| `--discount`          | The discount factor to use. It is set to `0` by default since we are in a contextual bandit setup                                                                                                           |
| `--noise_clip`        | The range to clip the target policy noise. The default value is set to `0.1`                                                                                                                                |
| `--lambda_critic`     | Lambda tradeoff for critic regularizer.                                                                                                                                                                     |
| `--folder`            | folder to save the results. The default directory is `./results/`                                                                                                                                           |
| `--human_feedback`    | The type of human feedback to provide. The choices are `none`, `action_recommendation`, `reward_manipuation`                                                                                                |
| `--feedback_interval` | The interval to query for human feedback. This can be any integer valued which determines the epocs after which human feedback would be obtained or it can be entropy based where `entropy` should be used. |
| `--entropy_threshold` | The entropy threshold value to be used this can be any positive float number                                                                                                                                |
| `--expert_accuracy`   | The accuracy of the expert level sould be a positve value between 0 and 1.                                                                                                                                  |
