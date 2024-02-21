import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn

import wandb
from accelerate import Accelerator
from time import time
from typing import List
from tqdm import tqdm

from backbones import SimCLR_series
from dataset import contrastive_dataset


# Log name
_exp_name = "Sample_" + str(time())


# SimCLR train agent
class SimCLR_trainer(object):
    def __init__(self, *args, **kwargs):
        self.conf = kwargs['conf']
        self.writer = SummaryWriter("./logs")
        # Boost training
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

    # Implement NT-Xent Loss
    # This implement make some changes, which is similar to log_softmax
    def _NT_Xent_Loss(self, zi: Tensor, zj: Tensor):
        # Concat the feature representations from two batches
        features_vector = torch.cat([zi, zj], dim=0)  # Tensor(batch_size * 2, features)

        # Compute cosine similarity for each feature representation
        features_vector = F.normalize(features_vector, dim=-1, p=2)
        similarity = torch.matmul(features_vector, features_vector.T) / self.conf.temperature

        # Creates a one-hot with broadcasting
        labels = torch.cat([torch.arange(self.conf.batch_size) for _ in range(self.conf.contrastive_channels)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # Tensor(batch_size * 2, batch_size * 2)

        # Remove self-similarity elements
        self_similarity = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~self_similarity].view(labels.shape[0], -1)              # Tensor(batch_size * 2, batch_size * 2 - 1)
        similarity = similarity[~self_similarity].view(similarity.shape[0], -1)  # Tensor(batch_size * 2, batch_size * 2 - 1)

        # select positives & negatives
        positives = similarity[labels.bool()].view(labels.shape[0], -1)       # Tensor(batch_size * 2, 1)
        negatives = similarity[~labels.bool()].view(similarity.shape[0], -1)  # Tensor(batch_size * 2, batch_size * 2 - 2)

        # Return the cross entropy
        # After concat(), the first element of logits represents positive pairs (zi, zj),
        # the others elements are negative pairs, so the targets of cross entropy is zeros, using the positive pairs as numerator
        logits = torch.concat([positives, negatives], dim=1)  # Tensor(batch_size * 2, batch_size * 2 - 1)
        targets = torch.zeros(labels.shape[0], dtype=torch.long).to(self.device)
        criterion = nn.CrossEntropyLoss()

        return criterion(logits, targets)

    # Pretrain process
    def SimCLR_pretrain(self, model_state_dict=None, optimizer_state_dict=None, scheduler_state_dict=None) -> None:
        # Dataset & Loader
        pretrain_dataset = contrastive_dataset(self.conf.data).get_dataset(self.conf.pretrain_dataset,
                                                                           self.conf.contrastive_channels)
        pretrain_loader = DataLoader(pretrain_dataset,
                                     batch_size=self.conf.batch_size,
                                     shuffle=True,
                                     num_workers=self.conf.workers,
                                     drop_last=True)

        # Pretrain model
        pretrain_model = SimCLR_series.SimCLR_v1(self.conf.backbone, self.conf.width_multiplier, self.conf.features)

        # Optimizer & Scheduler
        # Default optimizer is NAdam for small batch size, scheduler is CosineAnnealingLR(Following the paper)
        optimizer = torch.optim.NAdam(pretrain_model.parameters(), lr=self.conf.learning_rate,
                                      weight_decay=self.conf.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(pretrain_loader))

        # Load state dict (if not None)
        if model_state_dict is not None:
            pretrain_model.load_state_dict(torch.load(model_state_dict))
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(torch.load(optimizer_state_dict))
        if scheduler_state_dict is not None:
            scheduler.load_state_dict(torch.load(scheduler_state_dict))

        # Initialize wandb process to record messages
        if self.conf.wandb_key:
            proj_name = f'SimCLR v1 ResNet50×{self.conf.width_multiplier} - Pretrain in {self.conf.pretrain_dataset} - {self.conf.batch_size} Batches {self.conf.epochs} Epochs'
            wandb.init(project="SimCLR v1", name=proj_name, config={
                "n_epochs": self.conf.epochs, "batch_size": self.conf.batch_size,
                "learning_rate": self.conf.learning_rate})

        # use accelerator to boost training
        model, dataloader, optimizer, scheduler = self.accelerator.prepare(pretrain_model, pretrain_loader, optimizer,
                                                                           scheduler)
        # X-axis of SummaryWriter & wandb
        step = 0

        # ---------- Pretraining ----------
        model.train()
        for epoch in range(self.conf.epochs):
            zi_record: List[Tensor] = []
            zj_record: List[Tensor] = []
            accumulation_counter = 1
            train_loss = []

            for batch in tqdm(dataloader):
                # Get the feature vector after nonlinear projection head
                images, _ = batch
                zi = model(images[0])
                zj = model(images[1])

                # Loss calculation
                if self.conf.batch_accumulation == 1:
                    loss = self._NT_Xent_Loss(zi, zj)
                    optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    optimizer.step()
                    train_loss.append(loss)
                else:
                    zi_record.append(zi)
                    zj_record.append(zj)
                    if accumulation_counter % self.conf.batch_accumulation == 0:
                        loss = self._NT_Xent_Loss(torch.cat(zi_record, dim=0), torch.cat(zj_record, dim=0))
                        optimizer.zero_grad()
                        self.accelerator.backward(loss)
                        optimizer.step()
                        train_loss.append(loss)

                accumulation_counter += 1

            # Record loss in this epoch
            step += 1
            avg_train_loss = sum(train_loss) / len(train_loss)
            self.writer.add_scalar("Loss/Pre-Train", avg_train_loss, step)
            if self.conf.wandb_key:
                wandb.log({"Pre-Train/Loss": avg_train_loss}, step=step)
            print(f"[ Pre-train | {epoch + 1:03d}/{self.conf.epochs:03d} ] loss = {avg_train_loss:.5f}")

            # Warmup in the first 10 rounds, then activate scheduler
            if epoch >= 10:
                scheduler.step()

            # Save model & optimizer checkpoint
            if (epoch + 1) % self.conf.save_checkpoint == 0:
                self.accelerator.wait_for_everyone()
                print("Saving checkpoints...")
                self.accelerator.save(self.accelerator.unwrap_model(model).state_dict(),
                                      f"./CheckPoints/ResNet50_{self.conf.width_multiplier}/backbone_checkpoint_{epoch + 1}_{self.conf.width_multiplier}.pt")
                self.accelerator.save(optimizer.state_dict(),
                                      f"./CheckPoints/ResNet50_{self.conf.width_multiplier}/optimizer_checkpoint_{epoch + 1}_{self.conf.width_multiplier}.pt")
                self.accelerator.save(scheduler.state_dict(),
                                      f"./CheckPoints/ResNet50_{self.conf.width_multiplier}/scheduler_checkpoint_{epoch + 1}_{self.conf.width_multiplier}.pt")
                print("Completed!")

        # Release Connection to wandb
        if self.conf.wandb_key:
            wandb.finish()

        # Free Memory
        self.accelerator.free_memory()
        del pretrain_model, pretrain_loader, optimizer, scheduler

    # Fine-tune parse for downstream task
    def SimCLR_classifier_finetune(self, model_state_dict) -> None:
        # Define transform in fine-tune parse
        # About Resize & CenterCrop, if you pre-train the SimCLR model on other resolution imgs,
        # make sure the output dim of CenterCrop() matches the pre-train dim to avoid loss of accuracy
        finetune_tfm = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.CenterCrop((96, 96)),
            transforms.ToImage(),
            transforms.ToDtype(dtype=torch.float32, scale=True),
        ])

        # Split train & valid set by 9:1, using only 10% of the training data
        finetune_set = torchvision.datasets.CIFAR10(root='./default_datasets',
                                                    train=True,
                                                    transform=finetune_tfm,
                                                    download=True)
        train_set = Subset(finetune_set, range(0, int(len(finetune_set) * 0.09)))
        valid_set = Subset(finetune_set, range(int(len(finetune_set) * 0.09), int(len(finetune_set) * 0.1)))

        train_loader = DataLoader(train_set, batch_size=self.conf.finetune_batchsize, shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_set, batch_size=self.conf.finetune_batchsize, shuffle=False, num_workers=2)

        # Replace the former projection head with a linear classifier, then load the state dict
        finetune_model = SimCLR_series.SimCLR_v1(self.conf.backbone, self.conf.width_multiplier, self.conf.features)
        finetune_model.backbone.fc = nn.Linear(2048 * self.conf.width_multiplier if self.conf.backbone == 'simclr_resnet50' else 1, 10)

        pretrained_dict = torch.load(model_state_dict)
        finetune_model_dict = finetune_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in finetune_model_dict}

        finetune_model_dict.update(pretrained_dict)
        finetune_model.load_state_dict(finetune_model_dict)

        # Define optimizer & loss function, load to gpu by accelerator
        optimizer = torch.optim.NAdam(finetune_model.parameters(), lr=self.conf.finetune_lr)
        criterion = nn.CrossEntropyLoss()
        finetune_model, optimizer, train_loader, valid_loader, criterion = self.accelerator.prepare(finetune_model,
                                                                                                    optimizer,
                                                                                                    train_loader,
                                                                                                    valid_loader,
                                                                                                    criterion)

        # wandb initialize
        if self.conf.wandb_key:
            proj_name = f'SimCLR v1 ResNet50*{self.conf.width_multiplier} via {self.conf.epochs} pre-train epochs - Classification fine-tune ResNet50 in 10% CIFAR-10'
            wandb.init(project="SimCLR v1", name=proj_name, config={
                "n_epochs": self.conf.fine_tune_epochs, "batch_size": self.conf.finetune_batchsize,
                "learning_rate": self.conf.finetune_lr})

        # Start Fine-tune & Testing
        best_acc = 0
        _exp_name = "fine-tune checkpoint"
        for epoch in range(self.conf.fine_tune_epochs):
            # ---------- Training ----------
            finetune_model.train()
            # These are used to record information in training.
            train_loss = []
            train_accs = []
            # Iter the whole train loader
            for batch in tqdm(train_loader, disable=not self.accelerator.is_local_main_process):
                # 1. Sample a batch
                imgs, labels = batch
                # 2. Forward the data
                logits = finetune_model(imgs)
                # 3. Calculate the loss
                loss = criterion(logits, labels)
                # 4. Clean former gradients, calculate new gradients, clip the gradient for stable training & backward
                optimizer.zero_grad()
                self.accelerator.backward(loss)
                grad_norm = nn.utils.clip_grad_norm_(finetune_model.parameters(), max_norm=10)
                optimizer.step()
                # 5. Compute the accuracy for current batch.
                acc = self.accelerator.gather(logits.argmax(dim=-1)) == self.accelerator.gather(labels)
                # acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
                # 6. Record the loss and accuracy
                train_accs.append(acc.long().sum().item() / acc.shape[0])
                train_loss.append(loss.item())
            # Print & Record the whole training process
            avg_train_loss = sum(train_loss) / len(train_loss)
            avg_train_acc = sum(train_accs) / len(train_accs)
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
            self.writer.add_scalar('Acc/train', avg_train_acc, epoch)
            if self.conf.wandb_key:
                wandb.log({"Train/Loss": avg_train_loss, "Train/Acc": avg_train_acc})
            print(
                f"[ Train | {epoch + 1:03d}/{self.conf.fine_tune_epochs:03d} ] loss = {avg_train_loss:.5f}, acc = {avg_train_acc:.5f}")

            # ---------- Validation ----------
            finetune_model.eval()
            # These are used to record information in validation.
            valid_loss = []
            valid_accs = []
            # Iter the whole valid loader
            for batch in tqdm(valid_loader, disable=not self.accelerator.is_local_main_process):
                # 1. Sample a batch
                imgs, labels = batch
                # 2. Forward the data
                # Using torch.no_grad() accelerates the forward process.
                with torch.no_grad():
                    logits = finetune_model(imgs)
                # 3. Calculate the loss
                loss = criterion(logits, labels)
                # 4. Compute the accuracy for current batch.
                acc = self.accelerator.gather(logits.argmax(dim=-1)) == self.accelerator.gather(labels)
                # 5. Record the loss and accuracy
                valid_accs.append(acc.long().sum().item() / acc.shape[0])
                valid_loss.append(loss.item())
            # Print & Record the whole valid process
            avg_valid_loss = sum(valid_loss) / len(valid_loss)
            avg_valid_acc = sum(valid_accs) / len(valid_accs)
            self.writer.add_scalar('Loss/valid', avg_valid_loss, epoch)
            self.writer.add_scalar('Acc/valid', avg_valid_acc, epoch)
            if self.conf.wandb_key:
                wandb.log({"Valid/Loss": avg_valid_loss, "Valid/Acc": avg_valid_acc})
            print(
                f"[ Valid | {epoch + 1:03d}/{self.conf.fine_tune_epochs:03d} ] loss = {avg_valid_loss:.5f}, acc = {avg_valid_acc:.5f}")

            # Update logs
            if avg_valid_acc > best_acc:
                with open(f"./{_exp_name}_log.txt", "a"):
                    print(
                        f"[ Valid | {epoch + 1:03d}/{self.conf.fine_tune_epochs:03d} ] loss = {avg_valid_loss:.5f}, acc = {avg_valid_acc:.5f} -> best")
            else:
                with open(f"./{_exp_name}_log.txt", "a"):
                    print(
                        f"[ Valid | {epoch + 1:03d}/{self.conf.fine_tune_epochs:03d} ] loss = {avg_valid_loss:.5f}, acc = {avg_valid_acc:.5f}")

            # save backbones
            if avg_valid_acc > best_acc:
                print(f"Best model found at epoch {epoch}, saving model")
                self.accelerator.save(self.accelerator.unwrap_model(finetune_model).state_dict(),
                                      f"./ResNet50_{self.conf.width_multiplier}/{self.conf.epochs} epochs SimCLR_finetune_best.ckpt")
                best_acc = avg_valid_acc

            # Release Connection to wandb
        wandb.finish()

        # Release Memory
        self.accelerator.free_memory()
        del train_loader, valid_loader, train_set, valid_set, finetune_model

        # ---------- Testing ----------
        # Replace the former projection head with a linear classifier, then load the state dict
        model_best = SimCLR_series.SimCLR_v1(self.conf.backbone, self.conf.width_multiplier, self.conf.features)
        model_best.backbone.fc = nn.Linear(2048 * self.conf.width_multiplier if self.conf.backbone == 'simclr_resnet50' else 1, 10)

        finetune_best = torch.load("./ResNet50_{self.conf.width_multiplier}/{self.conf.epochs} epochs SimCLR_finetune_best.ckpt")
        model_best.load_state_dict(finetune_best)

        model_best.eval()
        # Test set
        test_set = torchvision.datasets.CIFAR10(root='./default_datasets',
                                                train=False,
                                                transform=finetune_tfm)
        test_loader = DataLoader(test_set, batch_size=self.conf.finetune_batchsize, shuffle=False, num_workers=2)
        model_best, test_loader = self.accelerator.prepare(model_best, test_loader)
        # Calculate the accuracy
        total, matches = 0, 0
        # Start testing
        with torch.no_grad():
            for data, ground_truth in tqdm(test_loader, disable=not self.accelerator.is_local_main_process):
                test_prediction = model_best(data)
                _, pred = torch.max(test_prediction, dim=-1)
                total += ground_truth.size(0)
                matches += (pred == ground_truth).sum().item()

        test_accuracy = matches / total
        print(f"Model's accuracy of the network on the 10000 test images: {test_accuracy:.5f}")
        self.accelerator.free_memory()
        del model_best, test_loader
