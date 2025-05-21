from tbp.monty.frameworks.environments.embodied_data import EnvironmentDataLoader

class DroneDataLoader(EnvironmentDataLoader):
    def __init__(self, dataset, motor_system, object_names, object_init_sampler):
        super().__init__(dataset, motor_system)
        self.object_names = object_names
        self.object_init_sampler = object_init_sampler
        self.object_params = self.object_init_sampler()
        self.current_object = 0
        self.n_objects = len(self.object_names)
        self.episodes = 0
        self.epochs = 0
        self.primary_target = None

    def pre_episode(self):
        super().pre_episode()
        self.reset_agent()

    def post_episode(self):
        super().post_episode()
        self.object_init_sampler.post_episode()

    def reset_agent(self):
        # Reset DroneAgent
        self.observation, proprioceptive_state = self.dataset.reset()
        motor_system_state = MotorSystemState(proprioceptive_state)
        self._counter = 0
        