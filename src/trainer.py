import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score


def room_invariant_contrastive_loss(latent_repr, pedal_labels, room_labels, temperature=0.07):
    """
    Apply contrastive loss by selecting pairs where:
    - If room acoustics are different but pedal values are the same, the latent representations should be close
    - If room acoustics are the same but pedal values are different, the latent representations should be far

    Args:
    - latent_repr: Tensor of shape (bs * seq_len, hidden_dim)
    - pedal_labels: Tensor of shape (bs * seq_len)
    - room_labels: Tensor of shape (bs * seq_len)
    - temperature: Temperature parameter for the softmax
    """
    bs = room_labels.shape[0]
    # latent_repr: (bs * seq_len, hidden_dim) -> (bs, seq_len, hidden_dim)
    # pedal_labels: (bs * seq_len) -> (bs, seq_len)
    latent_repr = latent_repr.view(bs, -1, latent_repr.shape[-1])
    pedal_labels = pedal_labels.view(bs, -1)

    print(latent_repr.shape, pedal_labels.shape, room_labels.shape) # (bs, seq_len, hidden_dim), (bs, seq_len), (bs, )
    print("room_labels", room_labels)

    # Compute the pairwise similarity matrix
    similarity = torch.matmul(latent_repr, latent_repr.permute(0, 2, 1)) / temperature
    similarity = similarity - torch.eye(similarity.size(1), device=latent_repr.device) * 1e9

    # Positive mask
    pedal_labels = pedal_labels.unsqueeze(2) == pedal_labels.unsqueeze(1)
    room_labels = room_labels.unsqueeze(2) == room_labels.unsqueeze(1)
    positive_mask = pedal_labels & room_labels

    # Negative mask
    negative_mask = ~positive_mask

    # Positive similarity
    positive_similarity = similarity[positive_mask]

    # Negative similarity
    negative_similarity = similarity[negative_mask]

    # Compute the loss
    loss = -torch.log(
        torch.exp(positive_similarity).sum()
        / (torch.exp(negative_similarity).sum() + torch.exp(positive_similarity).sum())
    )

    return loss


def pedal_contrastive_loss(latent_repr, pedal_labels, temperature=0.07):
    _, hidden_dim = latent_repr.shape
    latent_repr = latent_repr.view(-1, hidden_dim)  # Flatten for pairwise computation
    pedal_labels = pedal_labels.view(-1)

    similarity = (
        torch.matmul(latent_repr, latent_repr.T) / temperature
    )  # Cosine similarity
    similarity = (
        similarity - torch.eye(similarity.size(0), device=latent_repr.device) * 1e9
    )  # Mask self-similarity

    pedal_labels = pedal_labels.unsqueeze(1) == pedal_labels.unsqueeze(0)  # Positive mask
    positive_similarity = similarity[pedal_labels]
    negative_similarity = similarity[~pedal_labels]

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
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
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

    def train(self, pedal_value_ratio=0.7, pedal_onset_ratio=0.1, pedal_offset_ratio=0.1,
              room_ratio=0.1, contrastive_ratio=0.2):
        best_val_losses = [float("inf")]
        for epoch in range(self.num_train_epochs):
            print(f"Starting Epoch {epoch + 1}/{self.num_train_epochs}")
            train_loss = self.train_one_epoch(epoch, pedal_value_ratio, pedal_onset_ratio, pedal_offset_ratio,
                                              room_ratio, contrastive_ratio)
            val_loss, val_pedal_value_f1, val_pedal_on_f1, val_pedal_off_f1, val_room_f1 = self.validate(
                epoch, pedal_value_ratio, pedal_onset_ratio, pedal_offset_ratio, room_ratio, contrastive_ratio
            )
            print(
                f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Pedal Value F1: {val_pedal_value_f1:.4f}, Val Pedal Onset F1: {val_pedal_on_f1:.4f}, Val Pedal Offset F1: {val_pedal_off_f1:.4f}, Val Room F1: {val_room_f1:.4f}"
            )

            if (epoch + 1) % self.eval_epochs == 0 and epoch != 0:
                # Save the model if it has the best validation loss
                if (
                    val_loss < max(best_val_losses)
                    or len(self.best_checkpoints) < self.save_total_limit
                ):
                    self.save_best_model(val_loss, val_pedal_value_f1, epoch)

                    if len(best_val_losses) > self.save_total_limit:
                        remove_idx = best_val_losses.index(max(best_val_losses))
                        best_val_losses.pop(remove_idx)
                        os.remove(self.best_checkpoints[remove_idx])
                        self.best_checkpoints.pop(remove_idx)

                    best_val_losses.append(val_loss)

    def train_one_epoch(self, epoch, pedal_value_ratio=0.7, pedal_onset_ratio=0.1, pedal_offset_ratio=0.1,
                        room_ratio=0.1, contrastive_ratio=0.1, room_contrastive_ratio=0.1):
        self.model.train()
        total_loss = 0

        for batch_idx, (inputs, p_v_labels, p_on_labels, p_off_labels, room_labels, midi_ids, pedal_factors) in enumerate(self.train_dataloader):
            inputs, p_v_labels, p_on_labels, p_off_labels = inputs.to(self.device), p_v_labels.to(self.device), p_on_labels.to(self.device), p_off_labels.to(self.device)
            room_labels, midi_ids, pedal_factors = room_labels.to(self.device), midi_ids.to(self.device), pedal_factors.to(self.device)

            loss_mask = p_v_labels != -1
            p_v_logits, p_on_logits, p_off_logits, room_logits, latent_repr = self.model(inputs, loss_mask=loss_mask)

            # Apply loss mask
            p_v_labels = p_v_labels[loss_mask]
            p_on_labels = p_on_labels[loss_mask]
            p_off_labels = p_off_labels[loss_mask]
            p_v_logits = p_v_logits[loss_mask]
            p_on_logits = p_on_logits[loss_mask]
            p_off_logits = p_off_logits[loss_mask]
            latent_repr = latent_repr[loss_mask]

            # Pedal classification loss
            p_v_loss = self.criterion(p_v_logits, p_v_labels)
            p_on_loss = self.bce_criterion(p_on_logits.squeeze(1), p_on_labels)
            p_off_loss = self.bce_criterion(p_off_logits.squeeze(1), p_off_labels)

            # Room classification loss
            room_labels = room_labels.long()
            room_loss = self.criterion(room_logits, room_labels)

            # Original contrastive loss (pedal-based)
            contrastive_loss_value = pedal_contrastive_loss(latent_repr, p_v_labels)

            # # New contrastive loss (room-invariant learning)
            # room_contrastive_loss_value = room_invariant_contrastive_loss(latent_repr, p_v_labels, room_labels)

            # Total loss
            loss = (
                pedal_value_ratio * p_v_loss
                + pedal_onset_ratio * p_on_loss
                + pedal_offset_ratio * p_off_loss
                + room_ratio * room_loss
                + contrastive_ratio * contrastive_loss_value
                # + room_contrastive_ratio * room_contrastive_loss_value
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.logging_steps == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(self.train_dataloader)}, "
                    f"Pedal Value Loss: {p_v_loss.item():.4f}, "
                    f"Pedal Onset Loss: {p_on_loss.item():.4f}, "
                    f"Pedal Offset Loss: {p_off_loss.item():.4f}, "
                    f"Room Loss: {room_loss.item():.4f}, "
                    f"Contrastive Loss: {contrastive_loss_value.item():.4f}, "
                    # f"Room Contrastive Loss: {room_contrastive_loss_value.item():.4f}, "
                    f"Total Loss: {loss.item():.4f}"
                )

            # Log to TensorBoard
            global_step = epoch * len(self.train_dataloader) + batch_idx
            self.writer.add_scalar("Train/Pedal Value Loss", p_v_loss.item(), global_step)
            self.writer.add_scalar("Train/Pedal Onset Loss", p_on_loss.item(), global_step)
            self.writer.add_scalar("Train/Pedal Offset Loss", p_off_loss.item(), global_step)
            self.writer.add_scalar("Train/Room Loss", room_loss.item(), global_step)
            self.writer.add_scalar("Train/Contrastive Loss", contrastive_loss_value.item(), global_step)
            # self.writer.add_scalar("Train/Room Contrastive Loss", room_contrastive_loss_value.item(), global_step)
            self.writer.add_scalar("Train/Total Loss", loss.item(), global_step)

        return total_loss / len(self.train_dataloader)

    def validate(self, epoch, pedal_value_ratio=0.7, pedal_onset_ratio=0.1, pedal_offset_ratio=0.1,
                 room_ratio=0.1, contrastive_ratio=0.2, room_contrastive_ratio=0.1):
        self.model.eval()
        val_loss = 0.0
        total_pedal_value_loss = 0.0
        total_pedal_on_loss = 0.0
        total_pedal_off_loss = 0.0
        total_room_loss = 0.0
        total_contrastive_loss = 0.0
        total_room_contrastive_loss = 0.0
        pedal_value_f1s = []
        pedal_onset_f1s = []
        pedal_offset_f1s = []
        room_f1s = []

        with torch.no_grad():
            for inputs, p_v_labels, p_on_labels, p_off_labels, room_labels, midi_ids, pedal_factors in self.val_dataloader:
                inputs, p_v_labels, p_on_labels, p_off_labels = inputs.to(self.device), p_v_labels.to(self.device), p_on_labels.to(self.device), p_off_labels.to(self.device)
                room_labels, midi_ids, pedal_factors = room_labels.to(self.device), midi_ids.to(self.device), pedal_factors.to(self.device)

                loss_mask = p_v_labels != -1
                p_v_logits, p_on_logits, p_off_logits, room_logits, latent_repr = self.model(inputs, loss_mask=loss_mask)

                # Apply loss mask
                p_v_labels = p_v_labels[loss_mask]
                p_on_labels = p_on_labels[loss_mask]
                p_off_labels = p_off_labels[loss_mask]
                p_v_logits = p_v_logits[loss_mask]
                p_on_logits = p_on_logits[loss_mask]
                p_off_logits = p_off_logits[loss_mask]
                latent_repr = latent_repr[loss_mask]

                # Pedal classification loss
                pedal_value_loss = self.criterion(p_v_logits, p_v_labels)
                pedal_on_loss = self.bce_criterion(p_on_logits.squeeze(1), p_on_labels)
                pedal_off_loss = self.bce_criterion(p_off_logits.squeeze(1), p_off_labels)

                # Room classification loss
                room_labels = room_labels.long()
                room_loss = self.criterion(room_logits, room_labels)

                # Contrastive losses
                contrastive_loss_value = pedal_contrastive_loss(latent_repr, p_v_labels)
                # room_contrastive_loss_value = room_invariant_contrastive_loss(latent_repr, midi_ids, pedal_factors, room_labels)

                # Total loss
                loss = (
                    pedal_value_ratio * pedal_value_loss
                    + pedal_onset_ratio * pedal_on_loss
                    + pedal_offset_ratio * pedal_off_loss
                    + room_ratio * room_loss
                    + contrastive_ratio * contrastive_loss_value
                    # + room_contrastive_ratio * room_contrastive_loss_value
                )

                val_loss += loss.item()
                total_pedal_value_loss += pedal_value_loss.item()
                total_pedal_on_loss += pedal_on_loss.item()
                total_pedal_off_loss += pedal_off_loss.item()
                total_room_loss += room_loss.item()
                total_contrastive_loss += contrastive_loss_value.item()
                # total_room_contrastive_loss += room_contrastive_loss_value.item()

                # Measure pedal value prediction
                p_v_outputs = torch.softmax(p_v_logits, dim=-1)
                p_v_preds = torch.argmax(p_v_outputs, dim=-1)
                p_v_labels = p_v_labels.cpu().numpy()
                p_v_preds = p_v_preds.cpu().numpy()
                pedal_value_f1 = f1_score(p_v_labels, p_v_preds, average="weighted")
                pedal_value_f1s.append(pedal_value_f1)

                # Measure pedal onset prediction
                p_on_preds = torch.sigmoid(p_on_logits) > 0.3
                p_on_labels = p_on_labels.cpu().numpy()
                p_on_preds = p_on_preds.cpu().numpy()
                # print(p_on_labels[:1000])
                pedal_onset_f1 = f1_score(p_on_labels > 0.3, p_on_preds, average="binary")
                pedal_onset_f1s.append(pedal_onset_f1)

                # Measure pedal offset prediction
                p_off_preds = torch.sigmoid(p_off_logits) > 0.3
                p_off_labels = p_off_labels.cpu().numpy()
                p_off_preds = p_off_preds.cpu().numpy()
                pedal_offset_f1 = f1_score(p_off_labels > 0.3, p_off_preds, average="binary")
                pedal_offset_f1s.append(pedal_offset_f1)

                # Measure room prediction
                room_outputs = torch.softmax(room_logits, dim=-1)
                room_preds = torch.argmax(room_outputs, dim=-1)
                room_preds = room_preds.cpu().numpy()
                room_labels = room_labels.cpu().numpy()
                room_f1 = f1_score(room_labels, room_preds, average="weighted")
                room_f1s.append(room_f1)

        # calculate avg f1
        avg_pedal_value_f1 = sum(pedal_value_f1s) / len(pedal_value_f1s)
        avg_pedal_onset_f1 = sum(pedal_onset_f1s) / len(pedal_onset_f1s)
        avg_pedal_offset_f1 = sum(pedal_offset_f1s) / len(pedal_offset_f1s)
        avg_room_f1 = sum(room_f1s) / len(room_f1s)

        # Log to TensorBoard
        self.writer.add_scalar(
            "Validation/Total Loss", val_loss / len(self.val_dataloader), epoch
        )
        self.writer.add_scalar(
            "Validation/Pedal Value Loss",
            total_pedal_value_loss / len(self.val_dataloader),
            epoch,
        )
        self.writer.add_scalar(
            "Validation/Pedal Onset Loss",
            total_pedal_on_loss / len(self.val_dataloader),
            epoch,
        )
        self.writer.add_scalar(
            "Validation/Pedal Offset Loss",
            total_pedal_off_loss / len(self.val_dataloader),
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
        # self.writer.add_scalar(
        #     "Validation/Room Contrastive Loss",
        #     total_room_contrastive_loss / len(self.val_dataloader),
        #     epoch,
        # )
        self.writer.add_scalar("Validation/Pedal Value F1", avg_pedal_value_f1, epoch)
        self.writer.add_scalar("Validation/Pedal Onset F1", avg_pedal_onset_f1, epoch)
        self.writer.add_scalar("Validation/Pedal Offset F1", avg_pedal_offset_f1, epoch)
        self.writer.add_scalar("Validation/Room F1", avg_room_f1, epoch)

        return val_loss / len(self.val_dataloader), avg_pedal_value_f1, avg_pedal_onset_f1, avg_pedal_offset_f1, avg_room_f1

    def save_best_model(self, val_loss, val_pedal_f1, epoch):
        best_checkpoint_path = os.path.join(
            self.save_dir,
            f"model_epoch_{epoch + 1}_val_loss_{val_loss:.4f}_val_pedal_f1_{val_pedal_f1:.4f}.pt",
        )
        torch.save(self.model.state_dict(), best_checkpoint_path)
        self.best_checkpoints.append(best_checkpoint_path)

        print(f"Best model saved at {best_checkpoint_path}")
