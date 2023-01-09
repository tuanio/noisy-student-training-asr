from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm


class Experimenter:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        log_wandb: bool,
        project_name: str,
        model_name: str,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.log_wandb = log_wandb

        if self.log_wandb:
            self.init_wandb(project_name, model_name)

    def init_wandb(self, project_name: str, model_name: str):
        wandb.init(project=project_name, name=model_name)

    def train(self):

        size = len(self.train_dataloader)
        pbar = tqdm(self.train_dataloader, total=size)

        for batch in pbar:
            ...

            self.optimizer.zero_grad()

            output, length = self.model(...)

    def test(self, is_valid: bool = True):
        ...
