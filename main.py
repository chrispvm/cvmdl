import torch
import torch.optim as optim

from . import nets

# My libraries

# random_seed = 42
# torch.manual_seed(random_seed)
torch.backends.cudnn.enabled = False

# utils.print_six_mnist_images(train_loader)

from . import tasks
from . import experiments

task = tasks.MNISTTask()
# input_size = int(math.prod(task.task_input_shape))
# hidden_sizes = [128, 64]
# output_size = int(math.prod(task.task_output_shape))
# print("dataset:", task.data_set)
# print("shape: ", task.data_set[0].shape())
# breakpoint()

exp_manager = experiments.ExperimentManager(experiment_id="MNIST",
                                            suggested_architecture=nets.CvmMnistNet,
                                            suggested_optimizer=optim.SGD,
                                            Task=tasks.MNISTTask)

exp_manager.run_all_experiments()

my_learners = []
for i in range(1, 5):
    net = nets.FullyConnectedReLULayers(task.input_shape, task.output_shape, width=10, nr_hidden=i)
    _learner = exp_manager.new_basic_learner(f"LargeReLU(L={i})", net)
    my_learners.append(_learner)

exp_manager.run_experiments_on_learners(my_learners, "Testing if hidden layers influence loss significantly",
                                        plot_individual_results=False)

exp_manager.open_results_dir()

# ==============

# import scripts.XScripts.MNIST_script

assert torch.cuda.is_available() is False
