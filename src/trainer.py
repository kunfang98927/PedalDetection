import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os


def contrastive_loss(embeddings, labels, temperature=0.07):
    _, hidden_dim = embeddings.shape
    embeddings = embeddings.view(-1, hidden_dim)  # Flatten for pairwise computation
    labels = labels.view(-1)

    similarity = (
        torch.matmul(embeddings, embeddings.T) / temperature
    )  # Cosine similarity
    similarity = (
        similarity - torch.eye(similarity.size(0), device=embeddings.device) * 1e9
    )  # Mask self-similarity

    labels = labels.unsqueeze(1) == labels.unsqueeze(0)  # Positive mask
    positive_similarity = similarity[labels]
    negative_similarity = similarity[~labels]

    loss = -torch.log(
        torch.exp(positive_similarity).sum()
        / (torch.exp(negative_similarity).sum() + torch.exp(positive_similarity).sum())
    )
    return loss


class PedalTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        criterion,
        optimizer,
        scheduler,
        device="cuda",
        logging_steps=10,
        eval_epochs=5,
        save_total_limit=20,
        save_dir="checkpoints",
        num_train_epochs=100,
        train_batch_size=32,
        val_batch_size=32,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.writer = SummaryWriter()
        self.logging_steps = logging_steps
        self.eval_epochs = eval_epochs
        self.save_total_limit = save_total_limit
        self.save_dir = save_dir
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.best_checkpoints = []  # To keep track of the best checkpoints
        os.makedirs(save_dir, exist_ok=True)

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
        )

    def train(self, alpha=0.5, beta=0.5):
        best_val_losses = [float("inf")]
        for epoch in range(self.num_train_epochs):
            print(f"Starting Epoch {epoch + 1}/{self.num_train_epochs}")
            train_loss = self.train_one_epoch(epoch, alpha=alpha, beta=beta)
            val_loss, val_accuracy = self.validate(epoch)
            print(
                f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )

            if (epoch + 1) % self.eval_epochs == 0 and epoch != 0:
                # Save the model if it has the best validation loss
                if (
                    val_loss < max(best_val_losses)
                    or len(self.best_checkpoints) < self.save_total_limit
                ):
                    self.save_best_model(val_loss, val_accuracy, epoch)

                    if len(best_val_losses) > self.save_total_limit:
                        remove_idx = best_val_losses.index(max(best_val_losses))
                        best_val_losses.pop(remove_idx)
                        os.remove(self.best_checkpoints[remove_idx])
                        self.best_checkpoints.pop(remove_idx)

                    best_val_losses.append(val_loss)

    def train_one_epoch(self, epoch, alpha=0.5, beta=0.5):
        self.model.train()
        total_loss = 0

        for batch_idx, (inputs, labels, acoustic_settings) in enumerate(
            self.train_dataloader
        ):

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            class_logits, embeddings = self.model(inputs)

            # Only calculate classification loss for valid labels
            loss_mask = labels != -1
            labels = labels[loss_mask]
            class_logits = class_logits[loss_mask]
            embeddings = embeddings[loss_mask]

            classification_loss = self.criterion(class_logits, labels)
            contrastive_loss_value = contrastive_loss(embeddings, labels)

            loss = alpha * classification_loss + beta * contrastive_loss_value

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Log to TensorBoard every logging_steps
            global_step = epoch * len(self.train_dataloader) + batch_idx
            if batch_idx % self.logging_steps == 0:
                self.writer.add_scalar(
                    "Train/Classification Loss", classification_loss.item(), global_step
                )
                self.writer.add_scalar(
                    "Train/Contrastive Loss", contrastive_loss_value.item(), global_step
                )
                self.writer.add_scalar("Train/Total Loss", loss.item(), global_step)

        return total_loss / len(self.train_dataloader)

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        total_classification_loss = 0.0
        total_contrastive_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels, acoustic_settings in self.val_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                class_logits, embeddings = self.model(inputs)

                loss_mask = labels != -1
                labels = labels[loss_mask]
                class_logits = class_logits[loss_mask]
                embeddings = embeddings[loss_mask]

                classification_loss = self.criterion(class_logits, labels)
                contrastive_loss_value = contrastive_loss(embeddings, labels)

                loss = classification_loss + contrastive_loss_value
                val_loss += loss.item()
                total_classification_loss += classification_loss.item()
                total_contrastive_loss += contrastive_loss_value.item()

                outputs = torch.softmax(class_logits, dim=-1)
                predictions = torch.argmax(outputs, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.numel()

        accuracy = correct / total

        # Log to TensorBoard
        self.writer.add_scalar(
            "Validation/Total Loss", val_loss / len(self.val_dataloader), epoch
        )
        self.writer.add_scalar(
            "Validation/Classification Loss",
            total_classification_loss / len(self.val_dataloader),
            epoch,
        )
        self.writer.add_scalar(
            "Validation/Contrastive Loss",
            total_contrastive_loss / len(self.val_dataloader),
            epoch,
        )
        self.writer.add_scalar("Validation/Accuracy", accuracy, epoch)

        return val_loss / len(self.val_dataloader), accuracy

    def save_best_model(self, val_loss, val_accuracy, epoch):
        best_checkpoint_path = os.path.join(
            self.save_dir,
            f"model_epoch_{epoch + 1}_val_loss_{val_loss:.4f}_val_acc_{val_accuracy:.4f}.pt",
        )
        torch.save(self.model.state_dict(), best_checkpoint_path)
        self.best_checkpoints.append(best_checkpoint_path)

        print(f"Best model saved at {best_checkpoint_path}")
