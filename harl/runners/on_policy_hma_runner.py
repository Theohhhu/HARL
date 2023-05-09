"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner
import random


class OnPolicyHMARunner(OnPolicyBaseRunner):
    """Runner for on-policy HA algorithms."""

    def train(self):
        """Train the model."""
        actor_train_infos = []

        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.algo_args['train']['episode_length'],
                self.algo_args['train']['n_rollout_threads'],
                1,
            ),
            dtype=np.float32,
        )

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(
                self.critic_buffer.value_preds[:-1]
            )
        else:
            advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]

        # normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # if self.fixed_order:
        #     agent_order = list(range(self.num_agents))
        # else:
        #     agent_order = list(torch.randperm(self.num_agents).numpy())

        # assume the group is pre divided
        group_order = []
        group_num = 3
        group_interval = round(self.num_agents // group_num) + 1
        # group_order = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]

        for i in range(0, group_num):
            one_group_agent_ls = []
            for j in range(group_interval):
                agent_index = i * group_interval + j
                if agent_index >= self.num_agents:
                    break
                one_group_agent_ls.append(agent_index)
            group_order.append(one_group_agent_ls)
        random.shuffle(group_order)

        for agent_in_group in group_order:

            # update agent factor all at once
            for agent_id in agent_in_group:
                self.actor_buffer[agent_id].update_factor(factor)  # current actor save factor

            # way 1 centroid factor in one group
            # here we assume one agent is the centroid
            agent_id_centroid = random.choice(agent_in_group)

            # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
            available_actions = (
                None
                if self.actor_buffer[agent_id_centroid].available_actions is None
                else self.actor_buffer[agent_id_centroid]
                         .available_actions[:-1]
                         .reshape(-1, *self.actor_buffer[agent_id_centroid].available_actions.shape[2:])
            )

            old_actions_logprob, _, _ = self.actor[agent_id_centroid].evaluate_actions(
                self.actor_buffer[agent_id_centroid]
                    .obs[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id_centroid].obs.shape[2:]),
                self.actor_buffer[agent_id_centroid]
                    .rnn_states[0:1]
                    .reshape(-1, *self.actor_buffer[agent_id_centroid].rnn_states.shape[2:]),
                self.actor_buffer[agent_id_centroid].actions.reshape(
                    -1, *self.actor_buffer[agent_id_centroid].actions.shape[2:]
                ),
                self.actor_buffer[agent_id_centroid]
                    .masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id_centroid].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id_centroid]
                    .active_masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id_centroid].active_masks.shape[2:]),
            )

            # update param in shared style
            random.shuffle(agent_in_group)
            for agent_id in agent_in_group:
                for agent_buffer_id in agent_in_group:
                    if self.state_type == "EP":
                        actor_train_info = self.actor[agent_id].train(
                            self.actor_buffer[agent_buffer_id], advantages.copy(), "EP"
                        )
                    elif self.state_type == "FP":
                        actor_train_info = self.actor[agent_id].train(
                            self.actor_buffer[agent_buffer_id], advantages[:, :, agent_buffer_id].copy(), "FP"
                        )
                actor_train_infos.append(actor_train_info)

            # compute action log probs for updated agent
            new_actions_logprob, _, _ = self.actor[agent_id_centroid].evaluate_actions(
                self.actor_buffer[agent_id_centroid]
                    .obs[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id_centroid].obs.shape[2:]),
                self.actor_buffer[agent_id_centroid]
                    .rnn_states[0:1]
                    .reshape(-1, *self.actor_buffer[agent_id_centroid].rnn_states.shape[2:]),
                self.actor_buffer[agent_id_centroid].actions.reshape(
                    -1, *self.actor_buffer[agent_id_centroid].actions.shape[2:]
                ),
                self.actor_buffer[agent_id_centroid]
                    .masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id_centroid].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id_centroid]
                    .active_masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id_centroid].active_masks.shape[2:]),
            )

            # update factor for next agent
            factor = factor * _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args['train']['episode_length'],
                    self.algo_args['train']['n_rollout_threads'],
                    1,
                )
            )

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info
