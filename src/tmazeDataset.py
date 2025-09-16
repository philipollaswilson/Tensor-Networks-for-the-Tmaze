from pymdp.envs import TMazeEnv as TMaze
import torch


config = Dict(
    {
        "log_dir": "tmaze_%Y%m%d_%H%M%S",
        "environment": Dict(
            {
                "reward_probs": [1, 0],
            }
        ),
        "pool": Dict(
            {
                "sequence_length": 3,
                "rollouts": 5000,
                "random": False,
            }
        ),
        "optimizer": Dict({"lr": 1e-3}),
        "trainer": Dict(
            {
                "batch_size": 100,
                "epochs": 500,
                "save_epoch": 50,
            }
        ),
        "model": Dict(
            {
                "init_mode": "positive",
                "max_bond": 24,
                "cutoff": 0.03,
                "dtype": "torch.complex128",
            }
        ),
        "device": "cuda",
    }
)



def make_env(config):
    return 

def make_dataset(config: Dict):
    dtype = "torch.complex128"
    env = DictPacker(TMaze(Dict({"reward_probs": [1, 0],})))
    observation_map = MultiOneHotMap(env.num_obs)
    action_map = MultiOneHotMap(env.num_controls)

    # init the agent
    agent = RandomAgent(env.get_action_space())

    # init xp pool
    pool = RolloutPool(
        env,
        agent,
        sequence_length=config.pool.sequence_length,
        epoch_size=config.pool.rollouts,
    )
    train = MemoryPool()

    if config.pool.random:
        for i in range(config.pool.rollouts):
            sample = pool[i]
            for k, v in sample.items():
                if k == "observation":
                    sample[k] = observation_map(v).type(dtype)
                elif k == "action":
                    sample[k] = action_map(v).type(dtype)
            train.push_no_update(sample)
    else:
        for c0 in range(2):
            for a1 in range(4):
                p1 = a1
                rmin1 = 1 if a1 in [1, 2] else 0
                rmax1 = 3 if a1 in [1, 2] else 1
                for r1 in range(rmin1, rmax1):
                    for c1 in range(2):
                        for a2 in range(4):
                            p2 = a2
                            if a1 in [1, 2]:
                                p2 = a1
                            if p1 in [1, 2] and p2 in [1, 2]:
                                rmin2 = r1
                                rmax2 = r1 + 1
                            elif p1 == 3 and p2 in [1, 2]:
                                rmin2 = 2 - (c1 + p2) % 2
                                rmax2 = rmin2 + 1
                            elif p2 in [1, 2]:
                                rmin2 = 1
                                rmax2 = 3
                            else:
                                rmin2 = 0
                                rmax2 = 1
                            for r2 in range(rmin2, rmax2):
                                cmin2 = c1 if a1 == 3 and a2 == 3 else 0
                                cmax2 = c1 + 1 if a1 == 3 and a2 == 3 else 2
                                for c2 in range(cmin2, cmax2):
                                    sequence = TensorDict(
                                        {
                                            "action": torch.tensor(
                                                [[0, 0], [a1, 0], [a2, 0]]
                                            ),
                                            "observation": torch.tensor(
                                                [[0, 0, c0], [p1, r1, c1], [p2, r2, c2]]
                                            ),
                                        }
                                    )
                                    for k, v in sequence.items():
                                        if k == "observation":
                                            sequence[k] = observation_map(v).type(dtype)
                                        elif k == "action":
                                            sequence[k] = action_map(v).type(dtype)
                                    train.push_no_update(sequence)
    train._update_table()

    validate = MemoryPool(
        sequence_length=config.pool.sequence_length,
        sequence_stride=config.pool.sequence_length + 1,
    )
    for i in range(config.trainer.batch_size):
        sample = pool[i]
        for k, v in sample.items():
            if k == "observation":
                sample[k] = observation_map(v).type(dtype)
            elif k == "action":
                sample[k] = action_map(v).type(dtype)
        validate.push(sample)

    return train, validate

