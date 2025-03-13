import os
import torch
import time
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error

import functools

print = functools.partial(print, flush=True)


def room_invariant_contrastive_loss(
    latent_repr, pedal_labels, room_labels, temperature=0.07
):
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

    print(
        latent_repr.shape, pedal_labels.shape, room_labels.shape
    )  # (bs, seq_len, hidden_dim), (bs, seq_len), (bs, )
    print("room_labels", room_labels)

    # Compute the pairwise similarity matrix
    similarity = torch.matmul(latent_repr, latent_repr.permute(0, 2, 1)) / temperature
    similarity = (
        similarity - torch.eye(similarity.size(1), device=latent_repr.device) * 1e9
    )

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

    pedal_labels = pedal_labels.unsqueeze(1) == pedal_labels.unsqueeze(
        0
    )  # Positive mask
    positive_similarity = similarity[pedal_labels]
    negative_similarity = similarity[~pedal_labels]

    loss = -torch.log(
        torch.exp(positive_similarity).sum()
        / (torch.exp(negative_similarity).sum() + torch.exp(positive_similarity).sum())
    )
    return loss


class PedalTrainer2:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        device="cuda",
        logging_steps=10,
        eval_steps=100,
        eval_epochs=-1,
        save_total_limit=20,
        save_dir="checkpoints",
        num_train_epochs=100,
        val_label_bin_edges=[0, 11, 95, 128],
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.ce_criterion = torch.nn.CrossEntropyLoss()
        self.mse_criterion = torch.nn.MSELoss()
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.writer = SummaryWriter()
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.eval_epochs = eval_epochs
        self.save_total_limit = save_total_limit
        self.save_dir = save_dir
        self.num_train_epochs = num_train_epochs
        self.val_label_bin_edges = val_label_bin_edges
        self.best_checkpoints = []  # To keep track of the best checkpoints
        os.makedirs(save_dir, exist_ok=True)

    def train(
        self,
        global_pedal_ratio=0.1,
        pedal_value_ratio=0.5,
        pedal_onset_ratio=0.1,
        pedal_offset_ratio=0.1,
        room_ratio=0.1,
        contrastive_ratio=0.1,
        start_epoch=0,
    ):
        best_val_losses = [float("inf")]
        global_step = 0
        # Start from `start_epoch` instead of 0
        for epoch in range(start_epoch, self.num_train_epochs):
            print(f"Starting Epoch {epoch + 1}/{self.num_train_epochs}")
            train_loss, global_step, best_val_losses = self.train_one_epoch(
                epoch,
                global_step,
                best_val_losses,
                global_pedal_ratio,
                pedal_value_ratio,
                pedal_onset_ratio,
                pedal_offset_ratio,
                room_ratio,
                contrastive_ratio,
            )
            if self.eval_steps == -1 and self.eval_epochs != -1:
                (
                    val_loss,
                    val_global_pedal_v_mae,
                    val_global_pedal_v_mse,
                    val_global_pedal_v_f1,
                    val_pedal_value_mae,
                    val_pedal_value_mse,
                    val_pedal_value_f1,
                    val_pedal_on_f1,
                    val_pedal_off_f1,
                    val_room_f1,
                ) = self.validate(
                    epoch,
                    global_pedal_ratio,
                    pedal_value_ratio,
                    pedal_onset_ratio,
                    pedal_offset_ratio,
                    room_ratio,
                    contrastive_ratio,
                )
                if (epoch + 1) % self.eval_epochs == 0 and epoch != 0:
                    # Save the model if it has the best validation loss
                    if (
                        val_loss < max(best_val_losses)
                        or len(self.best_checkpoints) < self.save_total_limit
                    ):
                        self.save_best_model(
                            val_loss,
                            val_pedal_value_mae,
                            val_pedal_value_f1,
                            epoch,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                        )

                        if len(best_val_losses) > self.save_total_limit:
                            remove_idx = best_val_losses.index(max(best_val_losses))
                            remove_idx_in_best_checkpoints = None
                            for i, checkpoint in enumerate(self.best_checkpoints):
                                if f"val_loss_{best_val_losses[remove_idx]}" in checkpoint:
                                    remove_idx_in_best_checkpoints = i
                                    break
                            print(
                                f"Removing {self.best_checkpoints[remove_idx_in_best_checkpoints]} with loss {best_val_losses[remove_idx]}"
                            )
                            os.remove(
                                self.best_checkpoints[remove_idx_in_best_checkpoints]
                            )
                            best_val_losses.pop(remove_idx)
                            self.best_checkpoints.pop(remove_idx_in_best_checkpoints)
                        best_val_losses.append(val_loss)

    def train_one_epoch(
        self,
        epoch,
        global_step=0,
        best_val_losses=[float("inf")],
        global_pedal_ratio=0.5,
        pedal_value_ratio=0.7,
        pedal_onset_ratio=0.1,
        pedal_offset_ratio=0.1,
        room_ratio=0.1,
        contrastive_ratio=0.1,
        room_contrastive_ratio=0.1,
    ):
        self.model.train()
        total_loss = 0

        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Epoch {epoch+1}",
        )
        for batch_idx, (
            inputs,
            global_p_v_labels,
            p_v_labels,
            p_on_labels,
            p_off_labels,
            room_labels,
            midi_ids,
            pedal_factors,
        ) in pbar:
            # Move data to device
            inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels = (
                inputs.to(self.device),
                global_p_v_labels.to(self.device),
                p_v_labels.to(self.device),
                p_on_labels.to(self.device),
                p_off_labels.to(self.device),
            )
            room_labels, midi_ids, pedal_factors = (
                room_labels.to(self.device),
                midi_ids.to(self.device),
                pedal_factors.to(self.device),
            )

            (
                global_p_v_logits,
                p_v_logits,
                p_on_logits,
                p_off_logits,
                room_logits,
                latent_repr,
            ) = self.model(inputs, loss_mask=None)

            latent_repr = latent_repr.reshape(-1, latent_repr.shape[-1])

            # Pedal classification loss
            p_v_loss = self.mse_criterion(p_v_logits.squeeze(), p_v_labels)
            p_on_loss = self.bce_criterion(p_on_logits.squeeze(), p_on_labels)
            p_off_loss = self.bce_criterion(p_off_logits.squeeze(), p_off_labels)

            # Room classification loss
            room_labels = room_labels.long()
            room_loss = self.ce_criterion(room_logits.squeeze(), room_labels)

            # Global pedal mse loss
            global_p_v_loss = self.mse_criterion(
                global_p_v_logits.squeeze(), global_p_v_labels
            )

            # Original contrastive loss (pedal-based)
            contrastive_loss_value = pedal_contrastive_loss(latent_repr, p_v_labels)

            # # New contrastive loss (room-invariant learning)
            # room_contrastive_loss_value = room_invariant_contrastive_loss(latent_repr, p_v_labels, room_labels)

            # Total loss
            loss = (
                global_pedal_ratio * global_p_v_loss
                + pedal_value_ratio * p_v_loss
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
            global_step += 1

            if batch_idx % self.logging_steps == 0:
                # print(
                #     f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(self.train_dataloader)}, Loss: "
                #     f"glob_p_v: {global_p_v_loss.item():.4f}, "
                #     f"p_v: {p_v_loss.item():.4f}, "
                #     f"p_on: {p_on_loss.item():.4f}, "
                #     f"p_off: {p_off_loss.item():.4f}, "
                #     f"room: {room_loss.item():.4f}, "
                #     f"p_cntr: {contrastive_loss_value.item():.4f}, "
                #     # f"Room Contrastive Loss: {room_contrastive_loss_value.item():.4f}, "
                #     f"total: {loss.item():.4f}"
                # )
                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "glob_p_v": global_p_v_loss.item(),
                        "p_v": p_v_loss.item(),
                        "p_on": p_on_loss.item(),
                        "p_off": p_off_loss.item(),
                        "room": room_loss.item(),
                        "p_cntr": contrastive_loss_value.item(),
                    }
                )

                # Log to TensorBoard
                self.writer.add_scalar(
                    "Train/Global Pedal Loss", global_p_v_loss.item(), global_step
                )
                self.writer.add_scalar(
                    "Train/Pedal Value Loss", p_v_loss.item(), global_step
                )
                self.writer.add_scalar(
                    "Train/Pedal Onset Loss", p_on_loss.item(), global_step
                )
                self.writer.add_scalar(
                    "Train/Pedal Offset Loss", p_off_loss.item(), global_step
                )
                self.writer.add_scalar("Train/Room Loss", room_loss.item(), global_step)
                self.writer.add_scalar(
                    "Train/Pedal Contrastive Loss",
                    contrastive_loss_value.item(),
                    global_step,
                )
                # self.writer.add_scalar("Train/Room Contrastive Loss", room_contrastive_loss_value.item(), global_step)
                self.writer.add_scalar("Train/Total Loss", loss.item(), global_step)

            # Validate by step, not just at epoch end.
            if self.eval_steps != -1 and global_step % self.eval_steps == 0:
                (
                    val_loss,
                    val_global_pedal_v_mae,
                    val_global_pedal_v_mse,
                    val_global_pedal_v_f1,
                    val_pedal_value_mae,
                    val_pedal_value_mse,
                    val_pedal_value_f1,
                    val_pedal_on_f1,
                    val_pedal_off_f1,
                    val_room_f1,
                ) = self.validate(
                    epoch,
                    global_pedal_ratio,
                    pedal_value_ratio,
                    pedal_onset_ratio,
                    pedal_offset_ratio,
                    room_ratio,
                    contrastive_ratio,
                )
                # Save best model if conditions are met
                if (
                    val_loss < max(best_val_losses)
                    or len(self.best_checkpoints) < self.save_total_limit
                ):
                    self.save_best_model(
                        val_loss,
                        val_pedal_value_mae,
                        val_pedal_value_f1,
                        epoch,
                        global_step=global_step,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                    )
                    if len(best_val_losses) > self.save_total_limit:
                        remove_idx = best_val_losses.index(max(best_val_losses))
                        remove_idx_in_best_checkpoints = None
                        for i, checkpoint in enumerate(self.best_checkpoints):
                            if f"val_loss_{best_val_losses[remove_idx]}" in checkpoint:
                                remove_idx_in_best_checkpoints = i
                                break
                        print(
                            f"Removing {self.best_checkpoints[remove_idx_in_best_checkpoints]} with loss {best_val_losses[remove_idx]}"
                        )
                        os.remove(self.best_checkpoints[remove_idx_in_best_checkpoints])
                        best_val_losses.pop(remove_idx)
                        self.best_checkpoints.pop(remove_idx_in_best_checkpoints)
                    best_val_losses.append(val_loss)

        return total_loss / len(self.train_dataloader), global_step, best_val_losses

    def validate(
        self,
        epoch,
        global_pedal_ratio=0.1,
        pedal_value_ratio=0.5,
        pedal_onset_ratio=0.1,
        pedal_offset_ratio=0.1,
        room_ratio=0.1,
        contrastive_ratio=0.1,
        room_contrastive_ratio=0.0,
    ):
        self.model.eval()
        val_loss = 0.0

        total_global_p_v_loss = 0.0
        total_pedal_value_loss = 0.0
        total_pedal_on_loss = 0.0
        total_pedal_off_loss = 0.0
        total_room_loss = 0.0
        total_contrastive_loss = 0.0
        total_room_contrastive_loss = 0.0

        global_pedal_value_maes = []
        global_pedal_value_mses = []
        global_pedal_value_f1s = []
        pedal_value_maes = []
        pedal_value_mses = []
        pedal_value_f1s = []
        pedal_onset_mses = []
        pedal_onset_maes = []
        pedal_onset_f1s = []
        pedal_offset_mses = []
        pedal_offset_maes = []
        pedal_offset_f1s = []
        room_f1s = []

        with torch.no_grad():
            pbar = tqdm(
                self.val_dataloader,
                total=len(self.val_dataloader),
                desc=f"Validation Epoch {epoch+1}",
            )
            for (
                inputs,
                global_p_v_labels,
                p_v_labels,
                p_on_labels,
                p_off_labels,
                room_labels,
                midi_ids,
                pedal_factors,
            ) in pbar:
                # Move data to device
                inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels = (
                    inputs.to(self.device),
                    global_p_v_labels.to(self.device),
                    p_v_labels.to(self.device),
                    p_on_labels.to(self.device),
                    p_off_labels.to(self.device),
                )
                room_labels, midi_ids, pedal_factors = (
                    room_labels.to(self.device),
                    midi_ids.to(self.device),
                    pedal_factors.to(self.device),
                )

                (
                    global_p_v_logits,
                    p_v_logits,
                    p_on_logits,
                    p_off_logits,
                    room_logits,
                    latent_repr,
                ) = self.model(inputs, loss_mask=None)

                latent_repr = latent_repr.reshape(-1, latent_repr.shape[-1])

                # Pedal classification loss
                pedal_value_loss = self.mse_criterion(p_v_logits.squeeze(), p_v_labels)
                pedal_on_loss = self.bce_criterion(p_on_logits.squeeze(), p_on_labels)
                pedal_off_loss = self.bce_criterion(
                    p_off_logits.squeeze(), p_off_labels
                )

                # Room classification loss
                room_labels = room_labels.long()
                room_loss = self.ce_criterion(room_logits.squeeze(), room_labels)

                # Global pedal mse loss
                global_p_v_loss = self.mse_criterion(
                    global_p_v_logits.squeeze(), global_p_v_labels
                )

                # Contrastive losses
                contrastive_loss_value = pedal_contrastive_loss(latent_repr, p_v_labels)
                # # room_contrastive_loss_value = room_invariant_contrastive_loss(latent_repr, midi_ids, pedal_factors, room_labels)

                # Total loss
                loss = (
                    global_pedal_ratio * global_p_v_loss
                    + pedal_value_ratio * pedal_value_loss
                    + pedal_onset_ratio * pedal_on_loss
                    + pedal_offset_ratio * pedal_off_loss
                    + room_ratio * room_loss
                    + contrastive_ratio * contrastive_loss_value
                    # + room_contrastive_ratio * room_contrastive_loss_value
                )

                val_loss += loss.item()
                total_global_p_v_loss += global_p_v_loss.item()
                total_pedal_value_loss += pedal_value_loss.item()
                total_pedal_on_loss += pedal_on_loss.item()
                total_pedal_off_loss += pedal_off_loss.item()
                total_room_loss += room_loss.item()
                total_contrastive_loss += contrastive_loss_value.item()
                # total_room_contrastive_loss += room_contrastive_loss_value.item()

                # Measure room prediction
                room_outputs = torch.softmax(room_logits, dim=-1)
                room_preds = torch.argmax(room_outputs, dim=-1)
                room_preds = room_preds.cpu().numpy()
                room_labels = room_labels.cpu().numpy()
                room_f1 = f1_score(room_labels, room_preds, average="weighted")
                room_f1s.append(room_f1)

                # Measure pedal value prediction
                p_v_preds = p_v_logits.squeeze()
                p_v_labels = p_v_labels.cpu().numpy()
                p_v_preds = p_v_preds.cpu().numpy()
                p_v_labels = p_v_labels.flatten()
                p_v_preds = p_v_preds.flatten()

                pedal_value_mse = mean_squared_error(p_v_labels, p_v_preds)
                pedal_value_mae = mean_absolute_error(p_v_labels, p_v_preds)
                pedal_value_maes.append(pedal_value_mae)
                pedal_value_mses.append(pedal_value_mse)

                # for p_v_labels and p_v_preds, if < 11, then 0, if > 95, then 2, else 1
                p_v_labels = p_v_labels * 127
                p_v_preds = p_v_preds * 127
                p_v_labels = np.digitize(p_v_labels, self.val_label_bin_edges)
                p_v_preds = np.digitize(p_v_preds, self.val_label_bin_edges)

                # f1
                pedal_value_f1 = f1_score(p_v_labels, p_v_preds, average="weighted")
                pedal_value_f1s.append(pedal_value_f1)

                # Measure pedal onset prediction
                p_on_preds = torch.sigmoid(p_on_logits).squeeze()
                p_on_labels = p_on_labels.cpu().numpy()
                p_on_preds = p_on_preds.cpu().numpy()
                p_on_labels = p_on_labels.flatten()
                p_on_preds = p_on_preds.flatten()

                # pedal_on_mse = mean_squared_error(p_on_labels, p_on_preds)
                # pedal_on_mae = mean_absolute_error(p_on_labels, p_on_preds)
                # pedal_onset_maes.append(pedal_on_mae)
                # pedal_onset_mses.append(pedal_on_mse)
                p_on_preds = np.where(p_on_preds > 0.5, 1, 0)
                p_on_labels = np.where(p_on_labels > 0.2, 1, 0)
                pedal_on_f1 = f1_score(p_on_labels, p_on_preds, average="binary")
                pedal_onset_f1s.append(pedal_on_f1)

                # Measure pedal offset prediction
                p_off_preds = torch.sigmoid(p_off_logits).squeeze()
                p_off_labels = p_off_labels.cpu().numpy()
                p_off_preds = p_off_preds.cpu().numpy()
                p_off_labels = p_off_labels.flatten()
                p_off_preds = p_off_preds.flatten()

                # pedal_off_mse = mean_squared_error(p_off_labels, p_off_preds)
                # pedal_off_mae = mean_absolute_error(p_off_labels, p_off_preds)
                # pedal_offset_maes.append(pedal_off_mae)
                # pedal_offset_mses.append(pedal_off_mse)
                p_off_preds = np.where(p_off_preds > 0.5, 1, 0)
                p_off_labels = np.where(p_on_labels > 0.2, 1, 0)
                pedal_off_f1 = f1_score(p_off_labels, p_off_preds, average="binary")
                pedal_offset_f1s.append(pedal_off_f1)

                # Measure global pedal value prediction
                # global_p_v_outputs = torch.softmax(global_p_v_logits, dim=-1)
                global_p_v_preds = global_p_v_logits
                global_p_v_labels = global_p_v_labels.cpu().numpy()
                global_p_v_preds = global_p_v_preds.cpu().numpy()

                global_pedal_value_mse = mean_squared_error(
                    global_p_v_labels, global_p_v_preds
                )
                global_pedal_value_mae = mean_absolute_error(
                    global_p_v_labels, global_p_v_preds
                )
                global_pedal_value_maes.append(global_pedal_value_mae)
                global_pedal_value_mses.append(global_pedal_value_mse)

                # for global_p_v_labels and global_p_v_preds, if < 11, then 0, if > 95, then 2, else 1
                global_p_v_labels = global_p_v_labels * 127
                global_p_v_preds = global_p_v_preds * 127
                global_p_v_labels = np.digitize(
                    global_p_v_labels, self.val_label_bin_edges
                )
                global_p_v_preds = np.digitize(
                    global_p_v_preds, self.val_label_bin_edges
                )

                # f1
                global_pedal_value_f1 = f1_score(
                    global_p_v_labels, global_p_v_preds, average="weighted"
                )
                global_pedal_value_f1s.append(global_pedal_value_f1)

        # calculate avg f1
        avg_global_pedal_value_f1 = sum(global_pedal_value_f1s) / len(
            global_pedal_value_f1s
        )
        avg_pedal_value_f1 = sum(pedal_value_f1s) / len(pedal_value_f1s)
        # avg_pedal_onset_mse = sum(pedal_onset_mses) / len(pedal_onset_mses)
        # avg_pedal_onset_mae = sum(pedal_onset_maes) / len(pedal_onset_maes)
        avg_pedal_onset_f1 = sum(pedal_onset_f1s) / len(pedal_onset_f1s)
        # avg_pedal_offset_mse = sum(pedal_offset_mses) / len(pedal_offset_mses)
        # avg_pedal_offset_mae = sum(pedal_offset_maes) / len(pedal_offset_maes)
        avg_pedal_offset_f1 = sum(pedal_offset_f1s) / len(pedal_offset_f1s)
        avg_room_f1 = sum(room_f1s) / len(room_f1s)
        avg_global_pedal_value_mae = sum(global_pedal_value_maes) / len(
            global_pedal_value_maes
        )
        avg_global_pedal_value_mse = sum(global_pedal_value_mses) / len(
            global_pedal_value_mses
        )
        avg_pedal_value_mae = sum(pedal_value_maes) / len(pedal_value_maes)
        avg_pedal_value_mse = sum(pedal_value_mses) / len(pedal_value_mses)

        # Log to TensorBoard
        self.writer.add_scalar(
            "Validation/Total Loss", val_loss / len(self.val_dataloader), epoch
        )
        self.writer.add_scalar(
            "Validation/Global Pedal Value Loss",
            total_global_p_v_loss / len(self.val_dataloader),
            epoch,
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
        self.writer.add_scalar(
            "Validation/Global Pedal Value F1", avg_global_pedal_value_f1, epoch
        )
        self.writer.add_scalar("Validation/Pedal Value F1", avg_pedal_value_f1, epoch)
        # self.writer.add_scalar("Validation/Pedal Onset MSE", avg_pedal_onset_mse, epoch)
        # self.writer.add_scalar("Validation/Pedal Onset MAE", avg_pedal_onset_mae, epoch)
        self.writer.add_scalar("Validation/Pedal Onset F1", avg_pedal_onset_f1, epoch)
        # self.writer.add_scalar("Validation/Pedal Offset MSE", avg_pedal_offset_mse, epoch)
        # self.writer.add_scalar("Validation/Pedal Offset MAE", avg_pedal_offset_mae, epoch)
        self.writer.add_scalar("Validation/Pedal Offset F1", avg_pedal_offset_f1, epoch)
        self.writer.add_scalar("Validation/Room F1", avg_room_f1, epoch)
        self.writer.add_scalar(
            "Validation/Global Pedal Value MAE", avg_global_pedal_value_mae, epoch
        )
        self.writer.add_scalar(
            "Validation/Global Pedal Value MSE", avg_global_pedal_value_mse, epoch
        )
        self.writer.add_scalar("Validation/Pedal Value MAE", avg_pedal_value_mae, epoch)
        self.writer.add_scalar("Validation/Pedal Value MSE", avg_pedal_value_mse, epoch)

        pbar.set_postfix(
            {
                "val_loss": val_loss / len(self.val_dataloader),
                "glob_p_v_f1": avg_global_pedal_value_f1,
                "p_v_f1": avg_pedal_value_f1,
                "p_on_f1": avg_pedal_onset_f1,
                "p_off_f1": avg_pedal_offset_f1,
                "room_f1": avg_room_f1,
                "glob_p_v_mae": avg_global_pedal_value_mae,
                "glob_p_v_mse": avg_global_pedal_value_mse,
                "p_v_mae": avg_pedal_value_mae,
                "p_v_mse": avg_pedal_value_mse,
            }
        )

        return (
            val_loss / len(self.val_dataloader),
            avg_global_pedal_value_mae,
            avg_global_pedal_value_mse,
            avg_global_pedal_value_f1,
            avg_pedal_value_mae,
            avg_pedal_value_mse,
            avg_pedal_value_f1,
            # avg_pedal_onset_mae,
            # avg_pedal_onset_mse,
            avg_pedal_onset_f1,
            # avg_pedal_offset_mae,
            # avg_pedal_offset_mse,
            avg_pedal_offset_f1,
            avg_room_f1,
        )

    def save_best_model(
        self,
        val_loss,
        val_pedal_v_mae,
        val_pedal_v_f1,
        epoch,
        global_step=None,
        optimizer=None,
        scheduler=None,
    ):
        if global_step is not None:
            best_checkpoint_path = os.path.join(
                self.save_dir,
                f"model_epoch_{epoch + 1}_step_{global_step}_val_loss_{val_loss:.4f}_f1_{val_pedal_v_f1:.4f}_mae_{val_pedal_v_mae:.4f}.pt",
            )
        else:
            best_checkpoint_path = os.path.join(
                self.save_dir,
                f"model_epoch_{epoch + 1}_val_loss_{val_loss:.4f}_f1_{val_pedal_v_f1:.4f}_mae_{val_pedal_v_mae:.4f}.pt",
            )

        try:
            # Save model.module.state_dict() if using DataParallel.
            model_state = (
                self.model.module.state_dict()
                if hasattr(self.model, "module")
                else self.model.state_dict()
            )
            torch.save(
                {
                    "model": model_state,
                    "optimizer": optimizer.state_dict() if optimizer else None,
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "epoch": epoch,
                },
                best_checkpoint_path,
            )

            self.best_checkpoints.append(best_checkpoint_path)
            print(f"Best model saved at {best_checkpoint_path}")

        except RuntimeError as e:
            print(f"Failed to save checkpoint: {e}")
