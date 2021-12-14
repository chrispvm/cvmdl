
from gym import wrappers




class ReinforcementLearner(Learner):

    def __init__(self, name, net, optimizer, environment, store_vid=False, callbacks=None):
        super().__init__(name, net, optimizer, callbacks)
        self.env = self._orig_env = environment
        self.storing_video = store_vid
        self._monitoring = False
        self.rendering = False
        self.episodes_per_epoch = 100

    # Override the existing method
    def _do_all_batches(self):
        self._do_all_episodes()
        # self._with_events(self._do_batch, "batch")

    def _do_all_episodes(self):
        values = []
        for episode in range(1, self.episodes_per_epoch):
            self.c_episode = episode
            self.c_batch_size = 1
            self._with_events(self._do_episode, "episode")

        # plt.plot(values)
        # plt.show()

        self.env.close()

    def __repr__(self):
        return f"NAME: {self.name}\n NET: {self.net}\n OPTIMIZER: {self.optimizer}\n ENV: " \
               f"{self.env}\n CALLBACKS: {self.callbacks}\n"

    def showcase_performance(self):
        self._start_monitoring()
        self.rendering = True
        e = self.episodes_per_epoch
        self.episodes_per_epoch = 10
        self._do_all_episodes()
        self.rendering = False
        self._stop_monitoring()
        self.episodes_per_epoch = e

    def store_video(self, store_vid=True):
        self.storing_video = store_vid

    def _start_monitoring(self):
        self._orig_env = self.env
        self.env = self._monitor_env = wrappers.Monitor(self.env, "./gym-results", force=True)
        self._monitoring = True

    def _stop_monitoring(self):
        self.env = self._orig_env
        self._monitoring = False

    def get_video_path(self):
        return f'./gym-results/openaigym.video.{self._monitor_env.file_infix}.video000000.mp4'

    def show_video(self):
        return Video(self.get_video_path(), embed=True)

    # ==================================================================================================================
    # SET MODES
    def train(self):
        super().train()
        if self.storing_video and self._monitoring:
            self._stop_monitoring()

    def eval(self):
        super().eval()
        if self.storing_video and not self._monitoring:
            self._start_monitoring()


class REINFORCELearner(ReinforcementLearner):

    def _do_episode(self):
        observations = []
        actions = []
        action_probs = []
        rewards = []
        o = self.env.reset()
        observations.append(o)
        done = False
        step = 0
        while not done:
            step += 1
            # print(f"episode {episode}. Step {step}")
            # print("kaas")
            if self.rendering:
                # print("kaas")
                self.env.render()
            o = torch.Tensor([o])
            ad = self.net(o)[0]

            ad = torch.softmax(ad, dim=-1)
            a = random.choices(population=range(0, self.env.action_space.n), weights=list(ad))
            a = a[0]
            # breakpoint()
            o, r, done, info = self.env.step(a)
            # self.env.render(mode='rgb_array')
            observations.append(o)
            actions.append(a)
            action_probs.append(ad)
            rewards.append(r)
        # for i,
        # values = [sum(rewards[i:]) for i in range(1, len(rewards))]
        if self.training_mode:
            for i, a in enumerate(actions):
                ad = action_probs[i]
                # print(ad)

                log_a = torch.log(ad[a])
                value = sum(rewards[i:])
                self.set_loss(- value * log_a)
                self.backward()
            self.optim_step()
        self.c_value = sum(rewards)
        # values.append(value)
        # breakpoint()
