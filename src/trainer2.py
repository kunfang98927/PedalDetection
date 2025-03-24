import os
import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error

import functools

print = functools.partial(print, flush=True)


def safe_normalize(x):
    """Avoid divide-by-zero or NaN."""
    norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-6)
    return x / norm

def compute_room_contrastive_loss(anchor, positive, negative, 
                                  base_margin=0.1, layer_idx=0, total_layers=8, 
                                  use_cosine=True):
    """
    Contrastive loss for room robustness, designed for per-layer application.
    - Margin decays per layer
    - Safe normalization and clamping
    - Optional L2 or Cosine mode
    - Softplus for stability
    """
    # Margin decay based on depth
    margin = base_margin * ((layer_idx + 1) / total_layers)

    # print min & max for debugging
    if torch.isnan(anchor).any() or torch.isinf(anchor).any() or torch.isnan(positive).any() or torch.isinf(positive).any() or torch.isnan(negative).any() or torch.isinf(negative).any():
        print(f"0 - Layer {layer_idx}, anchor min: {anchor.min().item()}, anchor max: {anchor.max().item()}")
        print(f"0 - Layer {layer_idx}, positive min: {positive.min().item()}, positive max: {positive.max().item()}")
        print(f"0 - Layer {layer_idx}, negative min: {negative.min().item()}, negative max: {negative.max().item()}")

    # Clamp to prevent inf
    anchor = torch.clamp(anchor, min=-5.0, max=5.0)
    positive = torch.clamp(positive, min=-5.0, max=5.0)
    negative = torch.clamp(negative, min=-5.0, max=5.0)
    if torch.isnan(anchor).any() or torch.isinf(anchor).any() or \
    torch.isnan(positive).any() or torch.isinf(positive).any() or \
    torch.isnan(negative).any() or torch.isinf(negative).any():
        print(f"[Warning] NaN or Inf detected BEFORE loss calc at layer {layer_idx}")
        
        # Auto-fix
        anchor = torch.nan_to_num(anchor, nan=0.0, posinf=0.0, neginf=0.0)
        positive = torch.nan_to_num(positive, nan=0.0, posinf=0.0, neginf=0.0)
        negative = torch.nan_to_num(negative, nan=0.0, posinf=0.0, neginf=0.0)

    # print min & max for debugging
    if torch.isnan(anchor).any() or torch.isinf(anchor).any() or torch.isnan(positive).any() or torch.isinf(positive).any() or torch.isnan(negative).any() or torch.isinf(negative).any():
        print(f"1 - Layer {layer_idx}, anchor min: {anchor.min().item()}, anchor max: {anchor.max().item()}")
        print(f"1 - Layer {layer_idx}, positive min: {positive.min().item()}, positive max: {positive.max().item()}")
        print(f"1 - Layer {layer_idx}, negative min: {negative.min().item()}, negative max: {negative.max().item()}")

    # Safe normalization
    anchor = safe_normalize(anchor)
    positive = safe_normalize(positive)
    negative = safe_normalize(negative)

    # print min & max for debugging
    if torch.isnan(anchor).any() or torch.isinf(anchor).any() or torch.isnan(positive).any() or torch.isinf(positive).any() or torch.isnan(negative).any() or torch.isinf(negative).any():
        print(f"2 - Layer {layer_idx}, anchor min: {anchor.min().item()}, anchor max: {anchor.max().item()}")
        print(f"2 - Layer {layer_idx}, positive min: {positive.min().item()}, positive max: {positive.max().item()}")
        print(f"2 - Layer {layer_idx}, negative min: {negative.min().item()}, negative max: {negative.max().item()}")

    # Debug check for NaN / Inf before computation
    if torch.isnan(positive).any() or torch.isinf(positive).any() or torch.isnan(negative).any() or torch.isinf(negative).any() or torch.isnan(anchor).any() or torch.isinf(anchor).any():
        print(f"[Warning] NaN or Inf detected in positive at layer {layer_idx}")
        return torch.tensor(0.0, requires_grad=True, device=anchor.device)

    if use_cosine:
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=1)
        loss = F.softplus(neg_sim - pos_sim + margin).mean()

        # print(f"[DEBUG] Layer {layer_idx}, margin={margin:.4f}, neg_sim - pos_sim: {(neg_sim - pos_sim).mean().item():.4f}")
    else:
        pos_dist = F.pairwise_distance(anchor, positive, p=2, eps=1e-8)
        neg_dist = F.pairwise_distance(anchor, negative, p=2, eps=1e-8)
        loss = F.softplus(pos_dist - neg_dist + margin).mean()

        # print(f"[DEBUG] Layer {layer_idx}, margin={margin:.4f}, pos_dist - neg_dist: {(pos_dist - neg_dist).mean().item():.4f}")

    # Final NaN check
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"[Warning] Contrastive loss unstable at layer {layer_idx}, skipping. Loss={loss.item()}")
        return torch.tensor(0.0, requires_grad=True, device=anchor.device)

    return loss


def pedal_contrastive_loss(latent_repr, pedal_labels, temperature=0.07):
    _, hidden_dim = latent_repr.shape

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
        eval_steps=-1,
        eval_epochs=-1,
        save_total_limit=20,
        save_dir="checkpoints",
        num_train_epochs=100,
        val_label_bin_edges=[0, 11, 95, 128],
        log_dir="logs",
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.ce_criterion = torch.nn.CrossEntropyLoss()
        self.mse_criterion = torch.nn.MSELoss(reduction="mean")
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.writer = SummaryWriter(log_dir=log_dir)
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
        global_pedal_ratio=0.2,
        pedal_value_ratio=0.6,
        pedal_onset_ratio=0.1,
        pedal_offset_ratio=0.1,
        room_ratio=0.0,
        contrastive_ratio=0.0,
        room_contrastive_ratio=0.1,
        start_epoch=0,
        start_global_step=-1,
    ):
        best_val_losses = [float("inf")]
        global_step = 0 if start_global_step == -1 else start_global_step
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
                room_contrastive_ratio,
            )
            if self.eval_steps == -1 and self.eval_epochs != -1 and (epoch+1) % self.eval_epochs == 0 and epoch != 0:
                (
                    val_loss,
                    val_global_pedal_v_mae,
                    val_global_pedal_v_mse,
                    val_global_pedal_v_f1,
                    val_pedal_value_mae,
                    val_pedal_value_mse,
                    val_pedal_value_f1,
                    val_pedal_on_mae,
                    val_pedal_off_mae,
                    val_room_f1,
                ) = self.validate(
                    epoch,
                    global_step,
                    global_pedal_ratio,
                    pedal_value_ratio,
                    pedal_onset_ratio,
                    pedal_offset_ratio,
                    room_ratio,
                    contrastive_ratio,
                    room_contrastive_ratio,
                )
                # Save the model if it is the best
                if len(self.best_checkpoints) < self.save_total_limit:
                    self.save_best_model(
                        val_loss,
                        val_pedal_value_mae,
                        val_pedal_value_f1,
                        epoch,
                        global_step=global_step,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                    )
                    # First checkpoint, remove the default value of inf
                    if len(best_val_losses) == 1 and float("inf") in best_val_losses:
                        best_val_losses = [val_loss]
                    else:
                        best_val_losses.append(val_loss)
                else:
                    # Select the worst checkpoint to remove
                    remove_idx = best_val_losses.index(max(best_val_losses))
                    remove_idx_in_best_checkpoints = None
                    for i, checkpoint in enumerate(self.best_checkpoints):
                        remove_loss = best_val_losses[remove_idx]
                        # round to 4 decimal places
                        if f"val_loss_{remove_loss:.4f}" in checkpoint:
                            remove_idx_in_best_checkpoints = i
                            break
                    print(
                        f"Removing {self.best_checkpoints[remove_idx_in_best_checkpoints]} with loss {best_val_losses[remove_idx]}"
                    )
                    os.remove(self.best_checkpoints[remove_idx_in_best_checkpoints])
                    best_val_losses.pop(remove_idx)
                    self.best_checkpoints.pop(remove_idx_in_best_checkpoints)
                    self.save_best_model(
                        val_loss,
                        val_pedal_value_mae,
                        val_pedal_value_f1,
                        epoch,
                        global_step=global_step,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                    )
                    best_val_losses.append(val_loss)

    def forward_for_one_batch(self, inputs, global_p_v_labels, p_v_labels, 
                              p_on_labels, p_off_labels, loss_mask,
                              room_labels, midi_ids, pedal_factors):

        # Move data to device
        inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels, loss_mask = (
            inputs.to(self.device),
            global_p_v_labels.to(self.device),
            p_v_labels.to(self.device),
            p_on_labels.to(self.device),
            p_off_labels.to(self.device),
            loss_mask.to(self.device),
        )
        room_labels, midi_ids, pedal_factors = (
            room_labels.to(self.device),
            midi_ids.to(self.device),
            pedal_factors.to(self.device),
        )

        self.model.train()
        (
            global_p_v_logits,
            p_v_logits,
            p_on_logits,
            p_off_logits,
            room_logits,
            latent_repr,
            mean_latent_repr,
        ) = self.model(inputs, loss_mask=loss_mask)

        # Apply loss_mask
        p_v_labels = p_v_labels[loss_mask == 1]
        p_v_logits = p_v_logits[loss_mask == 1]
        p_on_labels = p_on_labels[loss_mask == 1]   
        p_on_logits = p_on_logits[loss_mask == 1]
        p_off_labels = p_off_labels[loss_mask == 1]
        p_off_logits = p_off_logits[loss_mask == 1]
        latent_repr = latent_repr[loss_mask == 1]

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
        global_p_v_loss = global_p_v_loss.sum() / global_p_v_labels.shape[0]

        # Original contrastive loss (pedal-based)
        contrastive_loss_value = pedal_contrastive_loss(latent_repr, p_v_labels)
    
        return global_p_v_loss, p_v_loss, p_on_loss, p_off_loss, room_loss, contrastive_loss_value, mean_latent_repr

    def train_one_epoch(
        self,
        epoch,
        global_step=0,
        best_val_losses=[float("inf")],
        global_pedal_ratio=0.2,
        pedal_value_ratio=0.6,
        pedal_onset_ratio=0.1,
        pedal_offset_ratio=0.1,
        room_ratio=0.0,
        contrastive_ratio=0.0,
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
            loss_mask,
            room_labels,
            midi_ids,
            pedal_factors,
            *pair_features,
        ) in pbar:
            
            # Forward pass
            (
                global_p_v_loss, 
                p_v_loss, 
                p_on_loss, 
                p_off_loss, 
                room_loss, 
                contrastive_loss_value, 
                mean_latent_repr
            ) = self.forward_for_one_batch(
                inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels, loss_mask,
                room_labels, midi_ids, pedal_factors
            )
            room_contrastive_loss_value = torch.tensor(0.0).to(self.device)
            # Total loss
            loss = (
                global_pedal_ratio * global_p_v_loss
                + pedal_value_ratio * p_v_loss
                + pedal_onset_ratio * p_on_loss
                + pedal_offset_ratio * p_off_loss
                + room_ratio * room_loss
                + contrastive_ratio * contrastive_loss_value
                + room_contrastive_ratio * room_contrastive_loss_value
            )

            # If room contrastive loss is enabled
            if room_contrastive_ratio > 0:
                positive_pairs = pair_features[:9]
                negative_pairs = pair_features[9:]
                # forward pass for positive and negative pairs
                (
                    positive_global_p_v_loss,
                    positive_p_v_loss,
                    positive_p_on_loss,
                    positive_p_off_loss,
                    positive_room_loss,
                    positive_contrastive_loss_value,
                    positive_mean_latent_repr,
                ) = self.forward_for_one_batch(*positive_pairs)
                (
                    negative_global_p_v_loss,
                    negative_p_v_loss,
                    negative_p_on_loss,
                    negative_p_off_loss,
                    negative_room_loss,
                    negative_contrastive_loss_value,
                    negative_mean_latent_repr,
                ) = self.forward_for_one_batch(*negative_pairs)

                for layer_id, (pos_repr, neg_repr, mean_repr) in enumerate(zip(positive_mean_latent_repr, 
                                                                             negative_mean_latent_repr, 
                                                                             mean_latent_repr)):
                    if layer_id < 4:
                        continue
                    room_contrastive_loss_value += compute_room_contrastive_loss(
                        mean_repr, pos_repr, neg_repr, layer_idx=layer_id
                    )
                room_contrastive_loss_value /= 4 # 8 layers
                # room_contrastive_loss_value = compute_room_contrastive_loss(
                #     mean_latent_repr[-1], positive_mean_latent_repr[-1], negative_mean_latent_repr[-1], layer_idx=7
                # )    

                loss += (
                    global_pedal_ratio * (positive_global_p_v_loss + negative_global_p_v_loss)
                    + pedal_value_ratio * (positive_p_v_loss + negative_p_v_loss)
                    + pedal_onset_ratio * (positive_p_on_loss + negative_p_on_loss)
                    + pedal_offset_ratio * (positive_p_off_loss + negative_p_off_loss)
                    + room_ratio * (positive_room_loss + negative_room_loss)
                    + contrastive_ratio * (positive_contrastive_loss_value + negative_contrastive_loss_value)
                )
                loss /= 3
                loss += room_contrastive_ratio * room_contrastive_loss_value

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if batch_idx % self.logging_steps == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(self.train_dataloader)}, Loss: "
                    f"glob_p_v: {global_p_v_loss.item():.4f}, "
                    f"p_v: {p_v_loss.item():.4f}, "
                    f"p_on: {p_on_loss.item():.4f}, "
                    f"p_off: {p_off_loss.item():.4f}, "
                    f"room: {room_loss.item():.4f}, "
                    f"p_cntr: {contrastive_loss_value.item():.4f}, "
                    f"Room Contrastive Loss: {room_contrastive_loss_value.item():.4f}, "
                    f"total: {loss.item():.4f}"
                )
                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "glob_p_v": global_p_v_loss.item(),
                        "p_v": p_v_loss.item(),
                        "p_on": p_on_loss.item(),
                        "p_off": p_off_loss.item(),
                        "room_cntr": room_contrastive_loss_value.item(),
                        "room": room_loss.item(),
                        "p_cntr": contrastive_loss_value.item(),
                    }
                )

                # Log to TensorBoard
                self.writer.add_scalars(
                    "Global Pedal Value Loss", {"Train": global_p_v_loss.item()}, global_step
                )
                self.writer.add_scalars(
                    "Pedal Value Loss", {"Train": p_v_loss.item()}, global_step
                )
                self.writer.add_scalars(
                    "Pedal Onset Loss", {"Train": p_on_loss.item()}, global_step
                )
                self.writer.add_scalars(
                    "Pedal Offset Loss", {"Train": p_off_loss.item()}, global_step
                )
                self.writer.add_scalars("Room Loss", {"Train": room_loss.item()}, global_step)
                self.writer.add_scalars(
                    "Pedal Contrastive Loss",
                    {"Train": contrastive_loss_value.item()},
                    global_step,
                )
                self.writer.add_scalars("Room Contrastive Loss", {"Train": room_contrastive_loss_value.item()}, global_step)
                self.writer.add_scalars("Total Loss", {"Train": loss.item()}, global_step)

                wandb.log(
                    {
                        "Global Pedal Loss/Train": global_p_v_loss.item(),
                        "Pedal Value Loss/Train": p_v_loss.item(),
                        "Pedal Onset Loss/Train": p_on_loss.item(),
                        "Pedal Offset Loss/Train": p_off_loss.item(),
                        "Room Contrastive Loss/Train": room_contrastive_loss_value.item(),
                        "Room Loss/Train": room_loss.item(),
                        "Pedal Contrastive Loss/Train": contrastive_loss_value.item(),
                        "Total Loss/Train": loss.item(),
                    }
                )

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
                    val_pedal_on_mae,
                    val_pedal_off_mae,
                    val_room_f1,
                ) = self.validate(
                    epoch,
                    global_step,
                    global_pedal_ratio,
                    pedal_value_ratio,
                    pedal_onset_ratio,
                    pedal_offset_ratio,
                    room_ratio,
                    contrastive_ratio,
                    room_contrastive_ratio,
                )
                # Save best model if conditions are met
                if len(self.best_checkpoints) < self.save_total_limit:
                    self.save_best_model(
                        val_loss,
                        val_pedal_value_mae,
                        val_pedal_value_f1,
                        epoch,
                        global_step=global_step,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                    )
                    if len(best_val_losses) == 1 and float("inf") in best_val_losses:
                        best_val_losses = [val_loss]
                    else:
                        best_val_losses.append(val_loss)
                else:
                    # Select the worst checkpoint to remove
                    remove_idx = best_val_losses.index(max(best_val_losses))
                    remove_idx_in_best_checkpoints = None
                    for i, checkpoint in enumerate(self.best_checkpoints):
                        remove_loss = best_val_losses[remove_idx]
                        # round to 4 decimal places
                        if f"val_loss_{remove_loss:.4f}" in checkpoint:
                            remove_idx_in_best_checkpoints = i
                            break
                    print(
                        f"Removing {self.best_checkpoints[remove_idx_in_best_checkpoints]} with loss {best_val_losses[remove_idx]}"
                    )
                    os.remove(self.best_checkpoints[remove_idx_in_best_checkpoints])
                    best_val_losses.pop(remove_idx)
                    self.best_checkpoints.pop(remove_idx_in_best_checkpoints)
                    self.save_best_model(
                        val_loss,
                        val_pedal_value_mae,
                        val_pedal_value_f1,
                        epoch,
                        global_step=global_step,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                    )
                    best_val_losses.append(val_loss)

        return total_loss / len(self.train_dataloader), global_step, best_val_losses

    def validate(
        self,
        epoch,
        global_step=-1,
        global_pedal_ratio=0.2,
        pedal_value_ratio=0.6,
        pedal_onset_ratio=0.1,
        pedal_offset_ratio=0.1,
        room_ratio=0.0,
        contrastive_ratio=0.0,
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
        pedal_onset_maes = []
        pedal_offset_maes = []
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
                loss_mask,
                room_labels,
                midi_ids,
                pedal_factors,
                *_,
            ) in pbar:
                # Move data to device
                inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels, loss_mask = (
                    inputs.to(self.device),
                    global_p_v_labels.to(self.device),
                    p_v_labels.to(self.device),
                    p_on_labels.to(self.device),
                    p_off_labels.to(self.device),
                    loss_mask.to(self.device),
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
                    mean_latent_repr,
                ) = self.model(inputs, loss_mask=loss_mask)

                # calculate valid frame number according to loss_mask
                # Apply loss_mask
                p_v_labels = p_v_labels[loss_mask == 1]
                p_v_logits = p_v_logits[loss_mask == 1]
                p_on_labels = p_on_labels[loss_mask == 1]
                p_on_logits = p_on_logits[loss_mask == 1]
                p_off_labels = p_off_labels[loss_mask == 1]
                p_off_logits = p_off_logits[loss_mask == 1]
                latent_repr = latent_repr[loss_mask == 1]

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
                global_p_v_loss = global_p_v_loss.sum() / global_p_v_labels.shape[0]

                # Contrastive losses
                contrastive_loss_value = pedal_contrastive_loss(latent_repr, p_v_labels)

                # Total loss
                loss = (
                    global_pedal_ratio * global_p_v_loss
                    + pedal_value_ratio * pedal_value_loss
                    + pedal_onset_ratio * pedal_on_loss
                    + pedal_offset_ratio * pedal_off_loss
                    + room_ratio * room_loss
                    + contrastive_ratio * contrastive_loss_value
                )

                val_loss += loss.item()
                total_global_p_v_loss += global_p_v_loss.item()
                total_pedal_value_loss += pedal_value_loss.item()
                total_pedal_on_loss += pedal_on_loss.item()
                total_pedal_off_loss += pedal_off_loss.item()
                total_room_loss += room_loss.item()
                total_contrastive_loss += contrastive_loss_value.item()
                # total_room_contrastive_loss += room_contrastive_loss_value.item()

                ################# Calculate metrics #################

                if room_ratio > 0:
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

                if pedal_onset_ratio > 0:
                    # Measure pedal onset prediction
                    p_on_preds = torch.sigmoid(p_on_logits).squeeze()
                    p_on_labels = p_on_labels.cpu().numpy()
                    p_on_preds = p_on_preds.cpu().numpy()
                    pedal_on_mae = mean_absolute_error(p_on_labels, p_on_preds)
                    pedal_onset_maes.append(pedal_on_mae)

                if pedal_offset_ratio > 0:
                    # Measure pedal offset prediction
                    p_off_preds = torch.sigmoid(p_off_logits).squeeze()
                    p_off_labels = p_off_labels.cpu().numpy()
                    p_off_preds = p_off_preds.cpu().numpy()
                    pedal_off_mae = mean_absolute_error(p_off_labels, p_off_preds)
                    pedal_offset_maes.append(pedal_off_mae)

                if global_pedal_ratio > 0:
                    # Measure global pedal value prediction
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
        ) if global_pedal_ratio > 0 else -1
        avg_pedal_value_f1 = sum(pedal_value_f1s) / len(pedal_value_f1s)
        avg_pedal_onset_mae = sum(pedal_onset_maes) / len(pedal_onset_maes) if pedal_onset_ratio > 0 else -1
        avg_pedal_offset_mae = sum(pedal_offset_maes) / len(pedal_offset_maes) if pedal_offset_ratio > 0 else -1
        avg_room_f1 = sum(room_f1s) / len(room_f1s) if room_ratio > 0 else -1
        avg_global_pedal_value_mae = sum(global_pedal_value_maes) / len(
            global_pedal_value_maes
        ) if global_pedal_ratio > 0 else -1
        avg_global_pedal_value_mse = sum(global_pedal_value_mses) / len(
            global_pedal_value_mses
        ) if global_pedal_ratio > 0 else -1
        avg_pedal_value_mae = sum(pedal_value_maes) / len(pedal_value_maes)
        avg_pedal_value_mse = sum(pedal_value_mses) / len(pedal_value_mses)

        log_step = global_step if global_step != -1 else epoch * len(
            self.train_dataloader
        )

        # Log to TensorBoard
        self.writer.add_scalars(
            "Total Loss", {"Val": val_loss / len(self.val_dataloader)}, log_step
        )
        self.writer.add_scalars(
            "Global Pedal Value Loss",
            {"Val": total_global_p_v_loss / len(self.val_dataloader)},
            log_step,
        )
        self.writer.add_scalars(
            "Pedal Value Loss",
            {"Val": total_pedal_value_loss / len(self.val_dataloader)},
            log_step,
        )
        self.writer.add_scalars(
            "Pedal Onset Loss",
            {"Val": total_pedal_on_loss / len(self.val_dataloader)},
            log_step,
        )
        self.writer.add_scalars(
            "Pedal Offset Loss",
            {"Val": total_pedal_off_loss / len(self.val_dataloader)},
            log_step,
        )
        self.writer.add_scalars(
            "Room Loss",
            {"Val": total_room_loss / len(self.val_dataloader)},
            log_step,
        )
        self.writer.add_scalars(
            "Pedal Contrastive Loss",
            {"Val": total_contrastive_loss / len(self.val_dataloader)},
            log_step,
        )
        # self.writer.add_scalars(
        #     "Room Contrastive Loss",
        #     total_room_contrastive_loss / len(self.val_dataloader),
        #     log_step,
        # )
        self.writer.add_scalar("Global Pedal Value F1", avg_global_pedal_value_f1, log_step)
        self.writer.add_scalar("Pedal Value F1", avg_pedal_value_f1, log_step)
        self.writer.add_scalar("Pedal Onset MAE", avg_pedal_onset_mae, log_step)
        self.writer.add_scalar("Pedal Offset MAE", avg_pedal_offset_mae, log_step)
        self.writer.add_scalar("Room F1", avg_room_f1, log_step)
        self.writer.add_scalar("Global Pedal Value MAE", avg_global_pedal_value_mae, log_step)
        self.writer.add_scalar("Global Pedal Value MSE", avg_global_pedal_value_mse, log_step)
        self.writer.add_scalar("Pedal Value MAE", avg_pedal_value_mae, log_step)
        self.writer.add_scalar("Pedal Value MSE", avg_pedal_value_mse, log_step)

        wandb.log(
            {
                "Total Loss/Val": val_loss / len(self.val_dataloader),
                "Global Pedal Value Loss/Val": total_global_p_v_loss / len(self.val_dataloader),
                "Pedal Value Loss/Val": total_pedal_value_loss / len(self.val_dataloader),
                "Pedal Onset Loss/Val": total_pedal_on_loss / len(self.val_dataloader),
                "Pedal Offset Loss/Val": total_pedal_off_loss / len(self.val_dataloader),
                "Room Loss/Val": total_room_loss / len(self.val_dataloader),
                "Pedal Contrastive Loss/Val": total_contrastive_loss / len(self.val_dataloader),
                # "Room Contrastive Loss": total_room_contrastive_loss / len(self.val_dataloader),
                "Global Pedal Value F1": avg_global_pedal_value_f1,
                "Pedal Value F1": avg_pedal_value_f1,
                "Pedal Onset MAE": avg_pedal_onset_mae,
                "Pedal Offset MAE": avg_pedal_offset_mae,
                "Room F1": avg_room_f1,
                "Global Pedal Value MAE": avg_global_pedal_value_mae,
                "Global Pedal Value MSE": avg_global_pedal_value_mse,
                "Pedal Value MAE": avg_pedal_value_mae,
                "Pedal Value MSE": avg_pedal_value_mse,
            }
        )

        pbar.set_postfix(
            {
                "val_loss": val_loss / len(self.val_dataloader),
                "glob_p_v_f1": avg_global_pedal_value_f1,
                "p_v_f1": avg_pedal_value_f1,
                "p_on_mae": avg_pedal_onset_mae,
                "p_off_mae": avg_pedal_offset_mae,
                "room_f1": avg_room_f1,
                "glob_p_v_mae": avg_global_pedal_value_mae,
                "glob_p_v_mse": avg_global_pedal_value_mse,
                "p_v_mae": avg_pedal_value_mae,
                "p_v_mse": avg_pedal_value_mse,
            }
        )
        print(
            f"Validation Loss: {val_loss / len(self.val_dataloader):.4f}, "
            f"glob_p_v_f1: {avg_global_pedal_value_f1:.4f}, "
            f"p_v_f1: {avg_pedal_value_f1:.4f}, "
            f"p_on_mae: {avg_pedal_onset_mae:.4f}, "
            f"p_off_mae: {avg_pedal_offset_mae:.4f}, "
            f"room_f1: {avg_room_f1:.4f}, "
            f"glob_p_v_mae: {avg_global_pedal_value_mae:.4f}, "
            f"glob_p_v_mse: {avg_global_pedal_value_mse:.4f}, "
            f"p_v_mae: {avg_pedal_value_mae:.4f}, "
            f"p_v_mse: {avg_pedal_value_mse:.4f}"
        )

        return (
            val_loss / len(self.val_dataloader),
            avg_global_pedal_value_mae,
            avg_global_pedal_value_mse,
            avg_global_pedal_value_f1,
            avg_pedal_value_mae,
            avg_pedal_value_mse,
            avg_pedal_value_f1,
            avg_pedal_onset_mae,
            avg_pedal_offset_mae,
            avg_room_f1,
        )

    def save_best_model(
        self,
        val_loss,
        val_pedal_v_mae,
        val_pedal_v_f1,
        epoch=None,
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
                    "global_step": global_step if global_step is not None else -1,
                },
                best_checkpoint_path,
            )
            # Save to wandb
            wandb.save(best_checkpoint_path)

            self.best_checkpoints.append(best_checkpoint_path)
            print(f"Best model saved at {best_checkpoint_path}")

        except RuntimeError as e:
            print(f"Failed to save checkpoint: {e}")
