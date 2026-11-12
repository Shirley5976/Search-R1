import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
import time
import numpy as np

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    # plan (P3 fork): siblings per forked trajectory, k* ~ Uniform{1..plan_kstar_max}, RNG seed
    plan_fork_m: int = 0
    plan_kstar_max: int = 4
    plan_seed: int = 0

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        # plan: k* is drawn before each rollout from its own RNG, independent of every sampled action
        self.plan_rng = np.random.RandomState(config.plan_seed)
        self.last_trace = None
        self.last_k_star = None
        self.last_siblings = None
        self.last_sib_records = None

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True,
                prompt_turn: torch.Tensor = None,
                turn_id: int = 0,
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists.
        plan: if prompt_turn is given, also carry a per-token decision-turn id (response tokens of this
        call get turn_id, observation and padding get 0) through the same reordering."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        if prompt_turn is None:
            return padded_tensor, padded_tensor_with_info
        turn_tensors = [prompt_turn, torch.where(response != pad_id, turn_id, 0).to(prompt_turn.dtype)]
        if info is not None:
            turn_tensors.append(torch.zeros_like(info, dtype=prompt_turn.dtype))
        padded_turn = torch.cat(turn_tensors, dim=1).gather(1, sorted_indices)
        return padded_tensor, padded_tensor_with_info, padded_turn

    def _update_right_side(self, right_side: Dict,
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None,
                          turn_id: int = 0) -> Dict:
        """Update right side state. plan: right_side['turn_ids'] (if present) tracks decision turns."""
        out = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                next_obs_ids,
                pad_to_left=False,
                prompt_turn=right_side.get('turn_ids'),
                turn_id=turn_id,
            )
        responses, responses_with_info_mask = out[0], out[1]
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        if effective_len > self.config.max_prompt_length:
            self.plan_truncated = True

        new_right = {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}
        if len(out) == 3:
            new_right['turn_ids'] = out[2][:, :max_len]
        return new_right

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop.

        plan additions: a per-trajectory trace of decision turns (self.last_trace) and per-token turn
        ids ('turn_ids'). In training with plan_fork_m = M > 0 (P3 / pilot), k* ~ Uniform{1..K} is
        drawn per trajectory before the rollout (self.last_k_star); at turn k* the M siblings are
        sampled in the same vLLM call and from the same prefix as the true action, their searches go
        in the same retrieval batch, and they are never continued (self.last_siblings,
        self.last_sib_records). Trajectories that end before k* are not forked.
        """
        bs = gen_batch.batch['input_ids'].shape[0]
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []],
                               'turn_ids': initial_input_ids[:, []]}

        fork_m = 0 if self.is_validation else int(self.config.plan_fork_m)
        k_star = None if self.is_validation else self.plan_rng.randint(1, self.config.plan_kstar_max + 1, size=bs)
        trace = [[] for _ in range(bs)]
        sib_parts, sib_records = [], []
        self.plan_truncated = False

        active_mask = torch.ones(bs, dtype=torch.bool)
        turns_stats = torch.ones(bs, dtype=torch.int)
        valid_action_stats = torch.zeros(bs, dtype=torch.int)
        valid_search_stats = torch.zeros(bs, dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            responses_ids, next_obs, dones, valid_action, is_search, meta_info = self._plan_turn(
                rollings, active_mask, step + 1, k_star, fork_m, original_right_side, trace, sib_parts, sib_records)

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)

            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids,
                turn_id=step + 1,
            )

        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            final_k = self.config.max_turns + 1
            responses_ids, _, dones, valid_action, is_search, meta_info = self._plan_turn(
                rollings, active_mask, final_k, k_star, fork_m, original_right_side, trace, sib_parts, sib_records,
                do_search=False)

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                turn_id=final_k,
            )

        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()

        print("ACTIVE_TRAJ_NUM:", active_num_list)

        self.last_trace = trace
        self.last_k_star = k_star
        self.last_sib_records = sib_records
        self.last_siblings = None
        width = original_right_side['responses'].shape[1]
        if sib_parts:
            width = max(width, max(rs['responses'].shape[1] for _, rs in sib_parts))
            parents = torch.cat([p for p, _ in sib_parts])
            sib_right = {key: torch.cat([self._pad_right(rs[key], width, key) for _, rs in sib_parts])
                         for key in original_right_side}
            sib_left = {'input_ids': original_left_side['input_ids'][parents]}
            self.last_siblings = self._compose_final_output(sib_left, sib_right, {})
        original_right_side = {key: self._pad_right(v, width, key) for key, v in original_right_side.items()}
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _pad_right(self, t: torch.Tensor, width: int, key: str) -> torch.Tensor:
        if t.shape[1] == width:
            return t
        # turn_ids use 0 outside response tokens; token tensors use the pad id
        value = 0 if key == 'turn_ids' else self.tokenizer.pad_token_id
        pad = torch.full((t.shape[0], width - t.shape[1]), value, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=1)

    def _plan_turn(self, rollings, active_mask, k, k_star, fork_m, right_side, trace, sib_parts, sib_records,
                   do_search=True):
        """One decision turn for the active trajectories, plus the M siblings of every trajectory
        whose k* equals k. Returns the original rows only; sibling state goes to sib_parts/records."""
        bs = active_mask.shape[0]
        active_idx = torch.nonzero(active_mask, as_tuple=False).view(-1).tolist()
        fork_parents = [i for i in active_idx if fork_m > 0 and k_star is not None and k_star[i] == k]
        sib_src = torch.tensor([p for p in fork_parents for _ in range(fork_m)], dtype=torch.long)
        n_act = len(active_idx)

        gen_in = {key: v[active_mask] for key, v in rollings.batch.items()}
        if len(sib_src):
            gen_in = {key: torch.cat([v, rollings.batch[key][sib_src]], dim=0) for key, v in gen_in.items()}
        gen_output = self._generate_with_gpu_padding(DataProto.from_dict(gen_in))
        meta_info = gen_output.meta_info

        all_ids, all_str = self._postprocess_responses(gen_output.batch['responses'])
        responses_ids, responses_str = self.tensor_fn._example_level_pad(all_ids[:n_act], all_str[:n_act], active_mask)
        sib_ids, sib_str = all_ids[n_act:], all_str[n_act:]
        n_sib = len(sib_str)

        # Execute originals and siblings together: one retrieval batch per turn
        exec_mask = torch.cat([active_mask, torch.ones(n_sib, dtype=torch.bool)])
        next_obs, dones, valid_action, is_search = self.execute_predictions(
            responses_str + sib_str, self.tokenizer.pad_token, exec_mask, do_search=do_search)
        actions, contents = self.postprocess_predictions(responses_str + sib_str)
        docs = self._last_docs

        for i in active_idx:
            trace[i].append({'k': k, 'action': actions[i], 'content': contents[i],
                             'executed': bool(is_search[i]) and do_search, 'docs': docs[i] or []})
        if n_sib:
            rs = {key: v[sib_src] for key, v in right_side.items()}
            rs = self._update_right_side(rs, sib_ids, None, turn_id=k)
            sib_parts.append((sib_src, rs))
            for s in range(n_sib):
                g = bs + s
                sib_records.append({'parent': int(sib_src[s]), 'j': s % fork_m + 1, 'k': k,
                                    'action': actions[g], 'content': contents[g],
                                    'executed': bool(is_search[g]) and do_search, 'docs': docs[g] or []})
        return (responses_ids, next_obs[:bs], dones[:bs], valid_action[:bs], is_search[:bs], meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            search_docs = list(self._last_search_docs)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])
            search_docs = [[] for _ in search_results]
        # plan: per-row list of retrieved passages (None if the row did not search), for the reader
        self._last_docs = [None] * len(cur_actions)

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    self._last_docs[i] = search_docs.pop(0)
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        t0 = time.perf_counter()
        results = self._batch_search(queries)['result'] if queries else []
        # plan: retrieval time per step, accumulated across turns (gen timing includes it)
        timing_raw = getattr(self, 'timing_raw', None)
        if timing_raw is not None:
            timing_raw['search'] = timing_raw.get('search', 0.0) + time.perf_counter() - t0
        
        self._last_search_docs = [self._passages2docs(result) for result in results]
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

    def _passages2docs(self, retrieval_result):
        """plan: the same passages as _passages2string, one string per document, for the reader."""
        docs = []
        for doc_item in retrieval_result:
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            docs.append(f"(Title: {title}) {text}")
        return docs
