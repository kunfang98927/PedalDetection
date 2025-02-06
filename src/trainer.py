import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score


def pedal_contrastive_loss(latent_repr, room_labels, temperature=0.07):
    _, hidden_dim = latent_repr.shape
    latent_repr = latent_repr.view(-1, hidden_dim)  # Flatten for pairwise computation
    room_labels = room_labels.view(-1)

    similarity = (
        torch.matmul(latent_repr, latent_repr.T) / temperature
    )  # Cosine similarity
    similarity = (
        similarity - torch.eye(similarity.size(0), device=latent_repr.device) * 1e9
    )  # Mask self-similarity

    room_labels = room_labels.unsqueeze(1) == room_labels.unsqueeze(0)  # Positive mask
    positive_similarity = similarity[room_labels]
    negative_similarity = similarity[~room_labels]

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

    def train(self, pedal_ratio=0.7, room_ratio=0.1, contrastive_ratio=0.2):
        best_val_losses = [float("inf")]
        for epoch in range(self.num_train_epochs):
            print(f"Starting Epoch {epoch + 1}/{self.num_train_epochs}")
            train_loss = self.train_one_epoch(epoch, pedal_ratio, room_ratio, contrastive_ratio)
            val_loss, val_pedal_f1, val_env_f1 = self.validate(epoch, pedal_ratio, room_ratio, contrastive_ratio)
            print(
                f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Pedal F1: {val_pedal_f1:.4f}, Val Room F1: {val_env_f1:.4f}"
            )

            if (epoch + 1) % self.eval_epochs == 0 and epoch != 0:
                # Save the model if it has the best validation loss
                if (
                    val_loss < max(best_val_losses)
                    or len(self.best_checkpoints) < self.save_total_limit
                ):
                    self.save_best_model(val_loss, val_pedal_f1, epoch)

                    if len(best_val_losses) > self.save_total_limit:
                        remove_idx = best_val_losses.index(max(best_val_losses))
                        best_val_losses.pop(remove_idx)
                        os.remove(self.best_checkpoints[remove_idx])
                        self.best_checkpoints.pop(remove_idx)

                    best_val_losses.append(val_loss)

    def train_one_epoch(self, epoch, pedal_ratio=0.7, room_ratio=0.1, contrastive_ratio=0.2):
        self.model.train()
        total_loss = 0

        for batch_idx, (inputs, labels, room_labels) in enumerate(
            self.train_dataloader
        ):
            inputs, labels, room_labels = inputs.to(self.device), labels.to(self.device), room_labels.to(self.device)
            loss_mask = labels != -1
            class_logits, room_logits, latent_repr = self.model(inputs, loss_mask=loss_mask)

            # Pedal classification loss
            labels = labels[loss_mask]
            class_logits = class_logits[loss_mask]
            latent_repr = latent_repr[loss_mask]
            classification_loss = self.criterion(class_logits, labels)

            # Room classification loss
            room_labels = room_labels.long()
            room_loss = self.criterion(room_logits, room_labels)

            # Contrastive loss
            contrastive_loss_value = pedal_contrastive_loss(latent_repr, labels)

            # Total loss
            loss = pedal_ratio * classification_loss + room_ratio * room_loss + contrastive_ratio * contrastive_loss_value

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.logging_steps == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(self.train_dataloader)}, Classification Loss: {classification_loss.item():.4f}, Room Loss: {room_loss.item():.4f}, Contrastive Loss: {contrastive_loss_value.item():.4f}, Total Loss: {loss.item():.4f}"
                )

            # Log to TensorBoard every logging_steps
            global_step = epoch * len(self.train_dataloader) + batch_idx
            if batch_idx % self.logging_steps == 0:
                self.writer.add_scalar(
                    "Train/Classification Loss", classification_loss.item(), global_step
                )
                self.writer.add_scalar(
                    "Train/Room Loss", room_loss.item(), global_step
                )
                self.writer.add_scalar(
                    "Train/Contrastive Loss", contrastive_loss_value.item(), global_step
                )
                self.writer.add_scalar("Train/Total Loss", loss.item(), global_step)

        return total_loss / len(self.train_dataloader)

    def validate(self, epoch, pedal_ratio=0.7, room_ratio=0.1, contrastive_ratio=0.2):
        self.model.eval()
        val_loss = 0.0
        total_classification_loss = 0.0
        total_room_loss = 0.0
        total_contrastive_loss = 0.0
        pedal_f1s = []
        env_f1s = []

        with torch.no_grad():
            for inputs, labels, room_labels in self.val_dataloader:
                inputs, labels, room_labels = inputs.to(self.device), labels.to(self.device), room_labels.to(self.device)
                loss_mask = labels != -1
                class_logits, room_logits, latent_repr = self.model(inputs, loss_mask=loss_mask)

                # Pedal classification loss
                labels = labels[loss_mask]
                class_logits = class_logits[loss_mask]
                latent_repr = latent_repr[loss_mask]
                classification_loss = self.criterion(class_logits, labels)

                # Room classification loss
                room_labels = room_labels.long()
                room_loss = self.criterion(room_logits, room_labels)

                # Contrastive loss
                contrastive_loss_value = pedal_contrastive_loss(latent_repr, labels)

                # Total loss
                loss = pedal_ratio * classification_loss + room_ratio * room_loss + contrastive_ratio * contrastive_loss_value

                val_loss += loss.item()
                total_classification_loss += classification_loss.item()
                total_room_loss += room_loss.item()
                total_contrastive_loss += contrastive_loss_value.item()

                # measure pedal prediction
                pedal_outputs = torch.softmax(class_logits, dim=-1)
                pedal_predictions = torch.argmax(pedal_outputs, dim=-1)
                labels = labels.cpu().numpy()
                pedal_predictions = pedal_predictions.cpu().numpy()
                pedal_f1 = f1_score(labels, pedal_predictions, average="weighted")
                pedal_f1s.append(pedal_f1)

                # measure room prediction
                env_outputs = torch.softmax(room_logits, dim=-1)
                env_predictions = torch.argmax(env_outputs, dim=-1)
                env_predictions = env_predictions.cpu().numpy()
                room_labels = room_labels.cpu().numpy()
                env_f1 = f1_score(room_labels, env_predictions, average="weighted")
                env_f1s.append(env_f1)

        # calculate avg f1
        avg_pedal_f1 = sum(pedal_f1s) / len(pedal_f1s)
        avg_env_f1 = sum(env_f1s) / len(env_f1s)

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
            "Validation/Room Loss",
            total_room_loss / len(self.val_dataloader),
            epoch,
        )
        self.writer.add_scalar(
            "Validation/Contrastive Loss",
            total_contrastive_loss / len(self.val_dataloader),
            epoch,
        )
        self.writer.add_scalar("Validation/Pedal F1", avg_pedal_f1, epoch)
        self.writer.add_scalar("Validation/Room F1", avg_env_f1, epoch)

        return val_loss / len(self.val_dataloader), avg_pedal_f1, avg_env_f1

    def save_best_model(self, val_loss, val_pedal_f1, epoch):
        best_checkpoint_path = os.path.join(
            self.save_dir,
            f"model_epoch_{epoch + 1}_val_loss_{val_loss:.4f}_val_pedal_f1_{val_pedal_f1:.4f}.pt",
        )
        torch.save(self.model.state_dict(), best_checkpoint_path)
        self.best_checkpoints.append(best_checkpoint_path)

        print(f"Best model saved at {best_checkpoint_path}")
