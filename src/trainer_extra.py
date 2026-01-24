import os
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error

import functools

print = functools.partial(print, flush=True)


class PedalTrainerBasic:
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
        val_label_bin_edges=[0, 64, 128],
        log_dir="logs",
        use_midi=False,
        use_pred_pedal=False
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
        self.use_midi = use_midi
        self.use_pred_pedal = use_pred_pedal
        os.makedirs(save_dir, exist_ok=True)

    def train(
        self,
        global_pedal_ratio=0.2,
        pedal_value_ratio=0.6,
        pedal_onset_ratio=0.1,
        pedal_offset_ratio=0.1,
        start_epoch=0,
        start_global_step=-1,
        step_in_epoch=0
    ):
        best_val_losses = [float("inf")]
        global_step = 0 if start_global_step == -1 else start_global_step
        # Start from `start_epoch` instead of 0
        for epoch in range(start_epoch, self.num_train_epochs):
            print(f"Starting Epoch {epoch + 1}/{self.num_train_epochs}")
            current_step_in_epoch = step_in_epoch if epoch == start_epoch else 0
            train_loss, global_step, best_val_losses = self.train_one_epoch(
                epoch,
                global_step=global_step,
                best_val_losses=best_val_losses,
                global_pedal_ratio=global_pedal_ratio,
                pedal_value_ratio=pedal_value_ratio,
                pedal_onset_ratio=pedal_onset_ratio,
                pedal_offset_ratio=pedal_offset_ratio,
                step_in_epoch=current_step_in_epoch
            )
            # Handle epoch-based schedulers (but NOT ReduceLROnPlateau)
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                scheduler_name = type(self.scheduler).__name__
                if scheduler_name in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR']:
                    self.scheduler.step()

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
                ) = self.validate(
                    epoch,
                    global_step,
                    global_pedal_ratio,
                    pedal_value_ratio,
                    pedal_onset_ratio,
                    pedal_offset_ratio,
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
                # Handle ReduceLROnPlateau AFTER validation
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    if type(self.scheduler).__name__ == 'ReduceLROnPlateau':
                        self.scheduler.step(val_loss)

    def forward_for_one_batch(self, inputs, midi_inputs, pedal_inputs, global_p_v_labels, p_v_labels, 
                              p_on_labels, p_off_labels, loss_mask):
        for i, msk in enumerate(loss_mask):
            if msk.sum() == 0:
                print("[Warning] Empty mask detected!")

        # Move data to device
        inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels, loss_mask = (
            inputs.to(self.device),
            global_p_v_labels.to(self.device),
            p_v_labels.to(self.device),
            p_on_labels.to(self.device),
            p_off_labels.to(self.device),
            loss_mask.to(self.device),
        )

        if self.use_midi and midi_inputs is not None:
            midi_inputs = midi_inputs.to(self.device)

        if self.use_pred_pedal and pedal_inputs is not None:
            pedal_inputs = pedal_inputs.to(self.device)

        self.model.train()

        # MODIFIED: Updated model forward call logic to handle all 4 modes
        if self.use_midi and self.use_pred_pedal:
            # Audio + MIDI + Predicted Pedal
            (
                global_p_v_logits,
                p_v_logits,
                p_on_logits,
                p_off_logits,
            ) = self.model(inputs, midi_inputs=midi_inputs, pred_pedal_inputs=pedal_inputs, loss_mask=loss_mask)
        elif self.use_midi:
            # Audio + MIDI only
            (
                global_p_v_logits,
                p_v_logits,
                p_on_logits,
                p_off_logits,
            ) = self.model(inputs, midi_inputs=midi_inputs, loss_mask=loss_mask)
        elif self.use_pred_pedal:
            # Audio + Predicted Pedal (new mode)
            (
                global_p_v_logits,
                p_v_logits,
                p_on_logits,
                p_off_logits,
            ) = self.model(inputs, pred_pedal_inputs=pedal_inputs, loss_mask=loss_mask)
        else:
            # Audio only
            (
                global_p_v_logits,
                p_v_logits,
                p_on_logits,
                p_off_logits,
            ) = self.model(inputs, loss_mask=loss_mask)

        # Apply loss_mask
        p_v_labels = p_v_labels[loss_mask == 1]
        p_v_logits = p_v_logits[loss_mask == 1]
        p_on_labels = p_on_labels[loss_mask == 1]   
        p_on_logits = p_on_logits[loss_mask == 1]
        p_off_labels = p_off_labels[loss_mask == 1]
        p_off_logits = p_off_logits[loss_mask == 1]

        # Pedal classification loss
        p_v_loss = self.mse_criterion(p_v_logits.squeeze(), p_v_labels)
        p_on_loss = self.bce_criterion(p_on_logits.squeeze(), p_on_labels)
        p_off_loss = self.bce_criterion(p_off_logits.squeeze(), p_off_labels)

        # Global pedal mse loss
        global_p_v_loss = self.mse_criterion(
            global_p_v_logits.squeeze(), global_p_v_labels
        )
        global_p_v_loss = global_p_v_loss.sum() / global_p_v_labels.shape[0]
    
        return global_p_v_loss, p_v_loss, p_on_loss, p_off_loss

    def train_one_epoch(
        self,
        epoch,
        global_step=0,
        best_val_losses=[float("inf")],
        global_pedal_ratio=0.2,
        pedal_value_ratio=0.6,
        pedal_onset_ratio=0.1,
        pedal_offset_ratio=0.1,
        step_in_epoch=0
    ):
        self.model.train()
        total_loss = 0

        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Epoch {epoch+1}",
        )

        # MODIFIED: Updated batch unpacking to handle all 4 modes
        for batch_idx, batch in pbar:
            # Initialize inputs
            midi_inputs = None
            pedal_inputs = None

            # Unpack batch based on the dataset configuration
            if self.use_midi and self.use_pred_pedal:
                # Audio + MIDI + Predicted Pedal (8 elements)
                (inputs, midi_inputs, pedal_inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels, loss_mask) = batch
            elif self.use_midi:
                # Audio + MIDI (7 elements)
                (inputs, midi_inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels, loss_mask) = batch
            elif self.use_pred_pedal:
                # Audio + Predicted Pedal (7 elements) - NEW MODE
                (inputs, pedal_inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels, loss_mask) = batch
            else:
                # Audio only (6 elements)
                (inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels, loss_mask) = batch
            
            if step_in_epoch > 0 and batch_idx <= step_in_epoch:
                # print(f"Skip batch {batch_idx} that has already been processed in the checkpoint.")
                continue
            elif step_in_epoch > 0 and batch_idx == step_in_epoch + 1:
                print(f"Processing batch {batch_idx} for the first time after resuming from checkpoint.")
                        
            # Forward pass
            (
                global_p_v_loss, 
                p_v_loss, 
                p_on_loss, 
                p_off_loss
            ) = self.forward_for_one_batch(
                inputs, midi_inputs, pedal_inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels, loss_mask
            )

            # Total loss
            loss = (
                global_pedal_ratio * global_p_v_loss
                + pedal_value_ratio * p_v_loss
                + pedal_onset_ratio * p_on_loss
                + pedal_offset_ratio * p_off_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Step-based scheduler updates (every batch)
            # Handle step-based schedulers
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                scheduler_name = type(self.scheduler).__name__
                if scheduler_name in ['OneCycleLR', 'CyclicLR', 'CosineAnnealingWarmRestarts']:
                    self.scheduler.step()

            total_loss += loss.item()
            global_step += 1

            if batch_idx % self.logging_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']

                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(self.train_dataloader)}, Loss: "
                    f"glob_p_v: {global_p_v_loss.item():.4f}, "
                    f"p_v: {p_v_loss.item():.4f}, "
                    f"p_on: {p_on_loss.item():.4f}, "
                    f"p_off: {p_off_loss.item():.4f}, "
                    f"total: {loss.item():.4f}"
                )
                
                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "glob_p_v": global_p_v_loss.item(),
                        "p_v": p_v_loss.item(),
                        "p_on": p_on_loss.item(),
                        "p_off": p_off_loss.item(),
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
                self.writer.add_scalars("Total Loss", {"Train": loss.item()}, global_step)

                self.writer.add_scalar("Learning Rate", current_lr, global_step)



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
                ) = self.validate(
                    epoch,
                    global_step,
                    global_pedal_ratio,
                    pedal_value_ratio,
                    pedal_onset_ratio,
                    pedal_offset_ratio,
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
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    if type(self.scheduler).__name__ == 'ReduceLROnPlateau':
                        self.scheduler.step(val_loss)

        return total_loss / len(self.train_dataloader), global_step, best_val_losses

    def validate(
        self,
        epoch,
        global_step=-1,
        global_pedal_ratio=0.2,
        pedal_value_ratio=0.6,
        pedal_onset_ratio=0.1,
        pedal_offset_ratio=0.1,
    ):
        self.model.eval()
        val_loss = 0.0

        total_global_p_v_loss = 0.0
        total_pedal_value_loss = 0.0
        total_pedal_on_loss = 0.0
        total_pedal_off_loss = 0.0

        global_pedal_value_maes = []
        global_pedal_value_mses = []
        global_pedal_value_f1s = []
        pedal_value_maes = []
        pedal_value_mses = []
        pedal_value_f1s = []
        pedal_onset_maes = []
        pedal_offset_maes = []

        with torch.no_grad():
            pbar = tqdm(
                self.val_dataloader,
                total=len(self.val_dataloader),
                desc=f"Validation Epoch {epoch+1}",
            )
            for batch in pbar:
                # Initialize inputs
                midi_inputs = None
                pedal_inputs = None

                # MODIFIED: Updated batch unpacking to handle all 4 modes
                if self.use_midi and self.use_pred_pedal:
                    # Audio + MIDI + Predicted Pedal (8 elements)
                    (inputs, midi_inputs, pedal_inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels, loss_mask) = batch
                elif self.use_midi:
                    # Audio + MIDI (7 elements)
                    (inputs, midi_inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels, loss_mask) = batch
                elif self.use_pred_pedal:
                    # Audio + Predicted Pedal (7 elements) - NEW MODE
                    (inputs, pedal_inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels, loss_mask) = batch
                else:
                    # Audio only (6 elements)
                    (inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels, loss_mask) = batch

                # Move data to device
                inputs, global_p_v_labels, p_v_labels, p_on_labels, p_off_labels, loss_mask = (
                    inputs.to(self.device),
                    global_p_v_labels.to(self.device),
                    p_v_labels.to(self.device),
                    p_on_labels.to(self.device),
                    p_off_labels.to(self.device),
                    loss_mask.to(self.device),
                )

                # Handle MIDI inputs
                if self.use_midi and midi_inputs is not None:
                    midi_inputs = midi_inputs.to(self.device)
                
                if self.use_pred_pedal and pedal_inputs is not None:
                    pedal_inputs = pedal_inputs.to(self.device)

                # MODIFIED: Updated model forward call logic to handle all 4 modes
                if self.use_midi and self.use_pred_pedal:
                    # Audio + MIDI + Predicted Pedal
                    (
                        global_p_v_logits,
                        p_v_logits,
                        p_on_logits,
                        p_off_logits,
                    ) = self.model(inputs, midi_inputs=midi_inputs, pred_pedal_inputs=pedal_inputs, loss_mask=loss_mask)
                elif self.use_midi:
                    # Audio + MIDI only
                    (
                        global_p_v_logits,
                        p_v_logits,
                        p_on_logits,
                        p_off_logits,
                    ) = self.model(inputs, midi_inputs=midi_inputs, loss_mask=loss_mask)
                elif self.use_pred_pedal:
                    # Audio + Predicted Pedal (new mode)
                    (
                        global_p_v_logits,
                        p_v_logits,
                        p_on_logits,
                        p_off_logits,
                    ) = self.model(inputs, pred_pedal_inputs=pedal_inputs, loss_mask=loss_mask)
                else:
                    # Audio only
                    (
                        global_p_v_logits,
                        p_v_logits,
                        p_on_logits,
                        p_off_logits,
                    ) = self.model(inputs, loss_mask=loss_mask)

                # calculate valid frame number according to loss_mask
                # Apply loss_mask
                p_v_labels = p_v_labels[loss_mask == 1]
                p_v_logits = p_v_logits[loss_mask == 1]
                p_on_labels = p_on_labels[loss_mask == 1]
                p_on_logits = p_on_logits[loss_mask == 1]
                p_off_labels = p_off_labels[loss_mask == 1]
                p_off_logits = p_off_logits[loss_mask == 1]

                # Pedal classification loss
                pedal_value_loss = self.mse_criterion(p_v_logits.squeeze(), p_v_labels)
                pedal_on_loss = self.bce_criterion(p_on_logits.squeeze(), p_on_labels)
                pedal_off_loss = self.bce_criterion(
                    p_off_logits.squeeze(), p_off_labels
                )

                # Global pedal mse loss
                global_p_v_loss = self.mse_criterion(
                    global_p_v_logits.squeeze(), global_p_v_labels
                )
                global_p_v_loss = global_p_v_loss.sum() / global_p_v_labels.shape[0]

                # Total loss
                loss = (
                    global_pedal_ratio * global_p_v_loss
                    + pedal_value_ratio * pedal_value_loss
                    + pedal_onset_ratio * pedal_on_loss
                    + pedal_offset_ratio * pedal_off_loss
                )

                val_loss += loss.item()
                total_global_p_v_loss += global_p_v_loss.item()
                total_pedal_value_loss += pedal_value_loss.item()
                total_pedal_on_loss += pedal_on_loss.item()
                total_pedal_off_loss += pedal_off_loss.item()

                ################# Calculate metrics #################

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
        self.writer.add_scalar("Global Pedal Value F1", avg_global_pedal_value_f1, log_step)
        self.writer.add_scalar("Pedal Value F1", avg_pedal_value_f1, log_step)
        self.writer.add_scalar("Pedal Onset MAE", avg_pedal_onset_mae, log_step)
        self.writer.add_scalar("Pedal Offset MAE", avg_pedal_offset_mae, log_step)
        self.writer.add_scalar("Global Pedal Value MAE", avg_global_pedal_value_mae, log_step)
        self.writer.add_scalar("Global Pedal Value MSE", avg_global_pedal_value_mse, log_step)
        self.writer.add_scalar("Pedal Value MAE", avg_pedal_value_mae, log_step)
        self.writer.add_scalar("Pedal Value MSE", avg_pedal_value_mse, log_step)


        pbar.set_postfix(
            {
                "val_loss": val_loss / len(self.val_dataloader),
                "glob_p_v_f1": avg_global_pedal_value_f1,
                "p_v_f1": avg_pedal_value_f1,
                "p_on_mae": avg_pedal_onset_mae,
                "p_off_mae": avg_pedal_offset_mae,
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
                    "use_midi": self.use_midi,
                    "use_pred_pedal": self.use_pred_pedal,  # ADDED: Save pedal configuration
                },
                best_checkpoint_path,
            )
            # Save to wandb
            # wandb.save(best_checkpoint_path)

            self.best_checkpoints.append(best_checkpoint_path)
            print(f"Best model saved at {best_checkpoint_path}")

        except RuntimeError as e:
            print(f"Failed to save checkpoint: {e}")