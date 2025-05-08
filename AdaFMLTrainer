from VirtualClient import VirtualClient
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from AdaFMLModel import AdaFMLModel
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import torch.optim as optim
from tqdm import tqdm
import copy
import torchvision
from config import Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AdaFMLTrainer:
    def __init__(self, config_instance: Config): 
        self.config = config_instance
        self.subsets, self.test_set = prepare_datasets(self.config)

        if len(self.subsets) != self.config.num_clients:
            print(f"Warning: Adjusting num_clients from {self.config.num_clients} to {len(self.subsets)}")
            self.config.num_clients = len(self.subsets)

        self.clients = [VirtualClient(client_id=i,
                                      data_loader=DataLoader(subset, batch_size=self.config.batch_size, shuffle=True,
                                                             num_workers=2, pin_memory=True, drop_last=True),
                                      config_instance=self.config)
                        for i, subset in enumerate(self.subsets)]

        if not self.clients: raise ValueError("No clients initialized.")
        self.config.num_clients = len(self.clients)

        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
        self.global_model = AdaFMLModel(self.config.dataset).to(device)

        self.alpha = nn.Parameter(torch.ones(self.config.num_clients, device=device) * 0.5)
        self.beta = nn.Parameter(torch.ones(self.config.num_clients, device=device) / self.config.num_clients)
        self.meta_optimizer = optim.AdamW([self.alpha, self.beta], lr=self.config.meta_lr_alpha)

        try:
            dummy_input_x = torch.randn(1, *self.global_model.input_example_shape).to(device)
            dummy_context = self.clients[0].get_context_features()

        except Exception as e:
            print(f"Could not add graph to TensorBoard: {e}")

        self.loss_records = {'task': [], 'time': [], 'meta': []}
        self.accuracy_history = []

        self.model_params_count = sum(p.numel() for p in self.global_model.parameters())
        self.time_normalizer = self._calibrate_time_normalizer()
        print(f"Calibrated time normalizer (based on expected time): {self.time_normalizer:.4f}")

    def _calibrate_time_normalizer(self):
        expected_time_samples = []
        for client in self.clients:
            exp_cp = client.calculate_expected_compute_time()
            exp_co = client.calculate_expected_communicate_time(self.model_params_count)
            if exp_cp != float('inf') and exp_co != float('inf') and exp_cp >= 0 and exp_co >= 0:
                expected_time_samples.append(exp_cp + exp_co)
            else:
                print(
                    f"Warning: Client {client.client_id} has invalid expected time (Cp={exp_cp:.4e}, Co={exp_co:.4e}). Excluding from normalization.")

        if not expected_time_samples:
            print("Warning: No valid expected time samples collected for normalization. Using default value 1.0.")
            return np.float32(1.0)

        normalizer = np.percentile(expected_time_samples, 95)
        min_normalizer_value = 1e-6
        final_normalizer = max(normalizer, min_normalizer_value)
        if normalizer < min_normalizer_value:
            print(
                f"Warning: Calculated time normalizer ({normalizer:.4e}) is very small. Clamping to {min_normalizer_value:.4e}.")
        return np.float32(final_normalizer)

    def _local_train(self, client_idx: int, global_model_state: Dict) -> Tuple[Dict, float, float, float]:

        client = self.clients[client_idx]
        model = AdaFMLModel(self.config.dataset).to(device)
        model.load_state_dict(global_model_state)
        model.train()

        task_losses, time_losses, total_losses = [], [], []
        optimizer = optim.AdamW(model.parameters(), lr=self.config.task_lr)
        local_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

        expected_cp = client.calculate_expected_compute_time()
        expected_co = client.calculate_expected_communicate_time(self.model_params_count)
        context_features = client.get_context_features()

        if expected_cp == float('inf') or expected_co == float('inf'):
            print(f"Warning: Client {client_idx} has infinite expected time. Skipping local training.")
            return None, 0.0, 0.0, 0.0

        expected_total_time_raw = expected_cp + expected_co
        if self.time_normalizer <= 1e-9:
            print(
                f"Warning: time_normalizer is too small ({self.time_normalizer:.4e}). Using raw time target for client {client_idx}.")
            normalized_expected_total_time = expected_total_time_raw
        else:
            normalized_expected_total_time = expected_total_time_raw / self.time_normalizer

        if not np.isfinite(normalized_expected_total_time):
            print(
                f"Warning: Normalized expected time target is not finite ({normalized_expected_total_time}) for client {client_idx}. Skipping local training.")
            return None, 0.0, 0.0, 0.0

        target_time_tensor_val = float(normalized_expected_total_time)

        for epoch in range(self.config.local_epochs):
            if len(client.loader) == 0: break
            for x, y in client.loader:
                if x.size(0) == 0: continue
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                current_context_features_batch = context_features.expand(x.size(0), -1)

                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    task_pred, time_pred = model(x, current_context_features_batch)
                    task_loss = nn.CrossEntropyLoss()(task_pred, y)
                    target_time_tensor = torch.full_like(time_pred, target_time_tensor_val, dtype=time_pred.dtype)
                    time_loss = nn.MSELoss()(time_pred, target_time_tensor)
                    alpha_factor = torch.sigmoid(self.alpha[client_idx].detach())
                    total_loss = (1 - alpha_factor) * task_loss + alpha_factor * time_loss

                optimizer.zero_grad()
                local_scaler.scale(total_loss).backward()
                local_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                local_scaler.step(optimizer)
                local_scaler.update()

                task_losses.append(task_loss.item())
                time_losses.append(time_loss.item())
                total_losses.append(total_loss.item())

        return model.state_dict(), np.mean(task_losses) if task_losses else 0.0, np.mean(
            time_losses) if time_losses else 0.0, np.mean(total_losses) if total_losses else 0.0

    def _aggregate_models(self, local_model_states: List[Dict], active_client_indices: List[int]):

        if not local_model_states or not active_client_indices: return
        valid_indices = [idx for idx in active_client_indices if idx < len(self.beta)]
        if not valid_indices: return
        beta_subset = self.beta[valid_indices]
        if beta_subset.numel() == 0: return
        beta_weights = torch.softmax(beta_subset, dim=0)

        aggregated_weights = {}
        global_model_param_names = self.global_model.state_dict().keys()
        valid_local_states = [local_model_states[active_client_indices.index(idx)] for idx in valid_indices if
                              idx in active_client_indices]
        if len(valid_local_states) != len(beta_weights): return

        for name in global_model_param_names:
            tensors_to_aggregate = [valid_local_states[i][name].float() * beta_weights[i] for i in
                                    range(len(valid_indices))]
            if tensors_to_aggregate:
                aggregated_weights[name] = torch.stack(tensors_to_aggregate).sum(dim=0)
            else:
                aggregated_weights[name] = self.global_model.state_dict()[name].clone()
        self.global_model.load_state_dict(aggregated_weights)

    def _meta_update(self, local_model_states: List[Dict], active_client_indices: List[int], round_idx: int):

        if not active_client_indices or not local_model_states: return 0.0
        valid_indices = [idx for idx in active_client_indices if idx < len(self.beta) and idx < len(self.alpha)]
        if not valid_indices: return 0.0
        valid_local_states = {idx: local_model_states[active_client_indices.index(idx)] for idx in valid_indices if
                              idx in active_client_indices}
        if len(valid_local_states) != len(valid_indices): return 0.0

        self.meta_optimizer.zero_grad()

        current_beta_params = self.beta[valid_indices]
        if current_beta_params.numel() == 0: return 0.0
        beta_weights_for_meta = torch.softmax(current_beta_params, dim=0)

        hat_theta_g_for_meta = {}
        template_state_dict = self.global_model.state_dict()
        for name, param_template in template_state_dict.items():
            hat_theta_g_for_meta[name] = torch.zeros_like(param_template, device=device, dtype=torch.float32)
        for i, client_idx_original in enumerate(valid_indices):
            for name in hat_theta_g_for_meta:
                if name in valid_local_states[client_idx_original]:
                    hat_theta_g_for_meta[name] += beta_weights_for_meta[i] * valid_local_states[client_idx_original][
                        name].detach().float()

        all_client_meta_losses_components = []
        for i, client_idx_original in enumerate(valid_indices):
            client = self.clients[client_idx_original]
            hat_theta_i_state = valid_local_states[client_idx_original]

            tilde_model = AdaFMLModel(self.config.dataset).to(device)
            alpha_i_factor = torch.sigmoid(self.alpha[client_idx_original])

            adapted_params = {}
            for name in hat_theta_i_state:
                if name in hat_theta_g_for_meta:
                    adapted_params[name] = alpha_i_factor * hat_theta_i_state[name].detach().float() + (
                                1 - alpha_i_factor) * hat_theta_g_for_meta[name]
                else:
                    adapted_params[name] = alpha_i_factor * hat_theta_i_state[name].detach().float()
            try:
                tilde_model.load_state_dict(adapted_params)
            except RuntimeError as e:
                print(f"Error loading state dict for tilde_model client {client_idx_original}: {e}"); continue
            tilde_model.train()

            client_task_loss_sum_meta = torch.tensor(0.0, device=device)
            client_time_loss_sum_meta = torch.tensor(0.0, device=device)
            num_meta_batches = 0
            if len(client.loader) == 0: continue

            expected_cp_meta = client.calculate_expected_compute_time()
            expected_co_meta = client.calculate_expected_communicate_time(self.model_params_count)
            context_features_meta = client.get_context_features()

            if expected_cp_meta == float('inf') or expected_co_meta == float('inf'): continue

            time_loss_target_raw_meta = expected_cp_meta + beta_weights_for_meta[i] * expected_co_meta
            if self.time_normalizer <= 1e-9:
                normalized_time_loss_target_meta = time_loss_target_raw_meta
            else:
                normalized_time_loss_target_meta = time_loss_target_raw_meta / self.time_normalizer

            if not torch.isfinite(normalized_time_loss_target_meta):
                print(
                    f"Warning: Meta normalized time target is not finite ({normalized_time_loss_target_meta}) for client {client_idx_original}. Skipping client.")
                continue

            target_time_tensor_meta_val = normalized_time_loss_target_meta.item()

            for batch_idx, (x_meta, y_meta) in enumerate(client.loader):
                if batch_idx >= 2: break
                if x_meta.size(0) == 0: continue
                x_meta = x_meta.to(device, non_blocking=True)
                y_meta = y_meta.to(device, non_blocking=True)
                current_context_features_batch_meta = context_features_meta.expand(x_meta.size(0), -1)

                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    task_pred_meta, time_pred_meta = tilde_model(x_meta, current_context_features_batch_meta)
                    loss_f_meta = nn.CrossEntropyLoss()(task_pred_meta, y_meta)
                    target_time_tensor_meta = torch.full_like(time_pred_meta, target_time_tensor_meta_val,
                                                              dtype=time_pred_meta.dtype)
                    loss_g_meta = nn.MSELoss()(time_pred_meta, target_time_tensor_meta)

                client_task_loss_sum_meta += loss_f_meta
                client_time_loss_sum_meta += loss_g_meta
                num_meta_batches += 1

            if num_meta_batches > 0:
                avg_client_task_meta_loss = client_task_loss_sum_meta / num_meta_batches
                avg_client_time_meta_loss = client_time_loss_sum_meta / num_meta_batches
                all_client_meta_losses_components.append((avg_client_task_meta_loss, avg_client_time_meta_loss))

        if not all_client_meta_losses_components: return 0.0

        stacked_meta_losses = torch.stack([task_l + time_l for task_l, time_l in all_client_meta_losses_components])
        total_meta_loss_sum_term = torch.mean(stacked_meta_losses)

        reg_alpha = self.config.lambda_alpha * torch.norm(self.alpha[valid_indices], p=2) ** 2
        reg_beta = self.config.lambda_beta * torch.norm(self.beta[valid_indices], p=2) ** 2
        total_meta_loss = total_meta_loss_sum_term + reg_alpha + reg_beta

        if device.type == 'cuda':
            self.scaler.scale(total_meta_loss).backward()
            self.scaler.unscale_(self.meta_optimizer)
            torch.nn.utils.clip_grad_norm_([self.alpha, self.beta], 1.0)
            self.scaler.step(self.meta_optimizer)
            self.scaler.update()
        else:
            total_meta_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.alpha, self.beta], 1.0)
            self.meta_optimizer.step()

        avg_meta_task_loss = torch.mean(torch.stack([comp[0] for comp in all_client_meta_losses_components])).item()
        avg_meta_time_loss = torch.mean(torch.stack([comp[1] for comp in all_client_meta_losses_components])).item()

        return total_meta_loss.item()

    def evaluate(self, round_idx: int, clients: List[VirtualClient], time_normalizer: float):

        self.global_model.eval()
        correct = 0
        total = 0
        total_test_task_loss = 0
        total_test_time_error = 0.0
        num_time_error_samples = 0

        test_loader = DataLoader(self.test_set, batch_size=self.config.batch_size, shuffle=False, num_workers=2,
                                 pin_memory=True)

        all_expected_times_raw = []
        for client in clients:
            exp_cp = client.calculate_expected_compute_time()
            exp_co = client.calculate_expected_communicate_time(self.model_params_count)
            if exp_cp != float('inf') and exp_co != float('inf'):
                all_expected_times_raw.append(exp_cp + exp_co)

        if all_expected_times_raw and time_normalizer > 1e-9:
            avg_expected_total_time_raw = np.mean(all_expected_times_raw)
            target_global_avg_exp_time_norm = avg_expected_total_time_raw / time_normalizer
            target_global_time_tensor_val = float(target_global_avg_exp_time_norm)
            calculate_global_time_error = True
        else:
            target_global_time_tensor_val = 0.0
            calculate_global_time_error = False
            print("Warning: Cannot calculate global time error target.")

        with torch.no_grad():
            for x, y in test_loader:
                if x.size(0) == 0: continue
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                dummy_context_eval = clients[0].get_context_features().expand(x.size(0), -1)

                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    task_pred, time_pred = self.global_model(x, dummy_context_eval)
                    task_loss = nn.CrossEntropyLoss()(task_pred, y)

                    if calculate_global_time_error:
                        target_global_time_tensor = torch.full_like(time_pred, target_global_time_tensor_val,
                                                                    dtype=time_pred.dtype)
                        time_error_batch = nn.MSELoss()(time_pred, target_global_time_tensor)
                        total_test_time_error += time_error_batch.item() * x.size(0)
                        num_time_error_samples += x.size(0)

                total_test_task_loss += task_loss.item()
                correct += (task_pred.argmax(1) == y).sum().item()
                total += y.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_test_task_loss = total_test_task_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        avg_global_time_error = total_test_time_error / num_time_error_samples if num_time_error_samples > 0 else 0.0

        self.global_model.train()
        return accuracy, avg_test_task_loss, avg_global_time_error

    def train(self):

        progress_bar = tqdm(range(self.config.num_rounds), desc="AdaFML Training (TruncExp Time)")
        for round_idx in progress_bar:
            current_global_model_state = copy.deepcopy(self.global_model.state_dict())

            active_client_indices_this_round = list(range(self.config.num_clients))
            local_model_states_collected = []
            avg_task_losses_round = []
            avg_time_losses_round = []
            avg_total_losses_round = []
            successful_client_indices = []

            for client_idx in active_client_indices_this_round:

                state_dict, task_l, time_l, total_l = self._local_train(client_idx, current_global_model_state)
                if state_dict is not None:
                    local_model_states_collected.append(state_dict)
                    successful_client_indices.append(client_idx)
                    if task_l is not None: avg_task_losses_round.append(task_l)
                    if time_l is not None: avg_time_losses_round.append(time_l)
                    if total_l is not None: avg_total_losses_round.append(total_l)
                else:
                    print(f"Warning: Client {client_idx} failed local training.")

            if local_model_states_collected:
                meta_loss_value = self._meta_update(local_model_states_collected, successful_client_indices, round_idx)
                self.loss_records['meta'].append(meta_loss_value)
            else:
                print(f"Round {round_idx + 1}: Skipping meta-update.")
                self.loss_records['meta'].append(0.0)

            if local_model_states_collected:
                self._aggregate_models(local_model_states_collected, successful_client_indices)
            else:
                print(f"Round {round_idx + 1}: Skipping model aggregation.")

            alpha_grad_norm = torch.norm(self.alpha.grad).item() if self.alpha.grad is not None else 0
            beta_grad_norm = torch.norm(self.beta.grad).item() if self.beta.grad is not None else 0

            if (round_idx + 1) % 2 == 0 or round_idx == self.config.num_rounds - 1:

                accuracy, test_task_loss, global_time_error = self.evaluate(round_idx, self.clients,
                                                                            self.time_normalizer)
                self.accuracy_history.append(accuracy)
                meta_loss_display = self.loss_records['meta'][-1] if self.loss_records['meta'] else 0.0
                progress_bar.set_postfix_str(
                    f"R: {round_idx + 1}, Acc: {accuracy:.2%}, TimeErr: {global_time_error:.4f}, MetaL: {meta_loss_display:.4f}"
                )
            else:
                meta_loss_display = self.loss_records['meta'][-1] if self.loss_records['meta'] else 0.0
                progress_bar.set_postfix_str(f"R: {round_idx + 1}, MetaL: {meta_loss_display:.4f}")
