# Configuration of Etical Trajectory Planning

There are several input files that configure the planning algorithm.
In the following, we will briefly describe every parameter that modifies the algorithms used in this repository. Exemplary settings are given by [`weights_ethical.json`](weights_ethical.json), [`weights_ego.json`](weights_ego.json) and [`weights_standard.json`](weights_standard.json).
## Cost Function Weights (weights.json)

[`weights.json`](weights.json) holds the paramters for the cost function of the trajectory planning. The different values arise from ethics as well as mobility and comfort. 

| Variable | Type | Description |
|----------|------|-------------|
| bayes | float | Cost weight for the Bayes principle |
| equality | float | Cost weight for the Equality principle |
| maximin | float | Cost weight for the Maximin principle |
| responsibility | float | Cost weight for the Responsibility principle |
| ego | float | Cost weight for the Ego principle |
| risk_cost | float | Cost weight for risk |
| visible_area | float | Cost weight for visible area (accounting for sensor occlusions) |
| lon_jerk | float | Cost weight for longitudinal jerk |
| lat_jerk | float | Cost weight for lateral jerk |
| velocity | float | Cost weight for target velocity |
| dist_to_global_path | float | Cost weight for distance to global path |
| travelled_dist | float | Cost weight for travelled distance |
| dist_to_goal_pos | float | Cost weight for distance to goal position |
| dist_to_lane_center | float | Cost weight for distance to center of the current lane |


## Risk parameters (risk.json)

[`risk.json`](risk.json) holds parameters for the risk and harm calculation.

| Variable | Type | Description |
|----------|------|-------------|
| harm_mode | str | |"log_reg" or "ref_speed" or "gidas" for different harm models |
| sensor_occlusion_model | str | Model to account for sensor occlusions in trajectory planning |
| occlusion_mode | bool | True to consider sensor occlusions in simulation and planning |
| ignore_angle | bool | True to ignore impact angle in harm model |
| sym_angle | bool | True to use symmetric angles along the longitudinal axis of the vehicle |
| reduced_angle_areas | bool | Reducing discretization of impact angle to 4 |
| trajectory_risk | str |  "max" or  "mean" to calculate with maxmium or mean cost per trajectory |
| max_acceptable_risk | float | Value between 0 and 1 to account for maximum acceptable risk |
| scale_factor_time | float | Scaling factor <1 over time with scale_factor^(time_step) |
| crash_angle_accuracy | int | Accuracy for crash angles in harm model |
| crash_angle_simplified | bool | True to use simplified crash angle calculation |
| figures: create_figures | bool | True to create and save extra figures |
| figures: number_plotted_trajectorie | int | Number of trajectories being plotted in these extra figures |
| risk_dashboard | bool | True to save risk visualizations as .png |
| collision_report | bool | True to save collision reports |



## Planning Parameters (planning.json)

[`planning.json`](planning.json) holds the planning parameters for the frenet planning approach used in this algorithm.

| Variable | Type | Description |
|----------|------|-------------|
| evaluation_settings:timing_enabled | bool | True to log detailed reports of execution times|
| evaluation_settings:show_visualization | bool | True to show live visualization |
| evaluation_settings:vehicle_type | str  | Vehicle type, e.g. "bmw_320i" - see CommonRoad for available vehicle models |
| frenet_settings:mode | str  | "risk" or "prediction" or "ground_truth" |
| frenet_parameters:t_list | list | List of planning horizons for frenet planning |
| frenet_parameters:v_list_generation_mode | str | Model to sample target velocities |
| frenet_parameters:n_v_samples | int | Number of disrecete velocties in trajectory sampling |
| frenet_parameters:d_list:d_max_abs | float | Single-sided width of the frenet space in m |
| frenet_parameters:d_list:n | int | Number of discrete paths along the width |
| frenet_parameters:dt | float | Vime step size in s|
| frenet_parameters:v_thr | float | Velocity threshold for switching to low velocity kinematic model |


## Predction Parameters (prediction.json)

Detailed information about the prediction network and its parameters can be found [here](https://github.com/TUMFTM/Wale-Net).


| Variable | Type | Description |
|----------|------|-------------|
|pred_config_path | str |Path to the config that was used for training, which holds relevant model informations |
|pred_model_path | str |Path to the model that should be used for prediction |
|gpu | str | String that specifies the system's GPU to run the training on|
|min_obs_length | int | First timestep a prediction step will be performed. Before this timestep the ground truth is predicted |
|on_pred_learn_method | str | Method for online learning | null, "switch_layer" or "parallel_head"|
|on_pred_horizon | list | List of integers that indicates when an online learning step is performed |
|on_lr | float |Learning rate for online learning |
|on_pred_learn_density | int | Only the i-th prediction will be used for online learning |
|online_layer | int |Number of online layers in the switch layer method |
|on_loss | str | Loss for online learning |
|on_optimizer | str | Optimizer for online learning |
|on_train_loss_threshold | float | Threshold for online learning |
