import logging
import math
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import autograd, distributions, nn, optim
from torch.nn import functional
from torch.utils.tensorboard import SummaryWriter

from .batch_buffer import BatchBuffer
from .nn_models import *
from .replay_buffer import PrioritizedReplayBuffer
from .utils import *
from .utils.prll_utils import LLMWrapper


class SAC_Base:
    def __init__(self,
                 obs_names: List[str],
                 obs_shapes: List[Tuple[int]],
                 d_action_sizes: List[int],
                 c_action_size: int,
                 model_abs_dir: Optional[Path],
                 device: Optional[str] = None,
                 ma_name: Optional[str] = None,
                 summary_path: Optional[str] = 'log',
                 train_mode: bool = True,
                 last_ckpt: Optional[str] = None,

                 nn_config: Optional[dict] = None,

                 nn=None,
                 seed: Optional[float] = None,
                 write_summary_per_step: float = 1e3,
                 save_model_per_step: float = 1e5,

                 use_replay_buffer: bool = True,
                 use_priority: bool = True,

                 ensemble_q_num: int = 2,
                 ensemble_q_sample: int = 2,

                 burn_in_step: int = 0,
                 n_step: int = 1,
                 seq_encoder: Optional[SEQ_ENCODER] = None,

                 batch_size: int = 256,
                 tau: float = 0.005,
                 update_target_per_step: int = 1,
                 init_log_alpha: float = -2.3,
                 use_auto_alpha: bool = True,
                 target_d_alpha: float = 0.98,
                 target_c_alpha: float = 1.,
                 d_policy_entropy_penalty: float = 0.5,

                 learning_rate: float = 3e-4,

                 gamma: float = 0.99,
                 v_lambda: float = 1.,
                 v_rho: float = 1.,
                 v_c: float = 1.,
                 clip_epsilon: float = 0.2,

                 discrete_dqn_like: bool = False,
                 use_n_step_is: bool = True,
                 siamese: Optional[SIAMESE] = None,
                 siamese_use_q: bool = False,
                 siamese_use_adaptive: bool = False,
                 use_prediction: bool = False,
                 transition_kl: float = 0.8,
                 use_extra_data: bool = True,
                 curiosity: Optional[CURIOSITY] = None,
                 curiosity_strength: float = 1.,
                 use_rnd: bool = False,
                 rnd_n_sample: int = 10,
                 use_normalization: bool = False,
                 action_noise: Optional[List[float]] = None,

                 replay_config: Optional[dict] = None):
        """
        obs_names: List of names of observations
        obs_shapes: List of dimensions of observations
        d_action_sizes: Dimensions of discrete actions
        c_action_size: Dimension of continuous actions
        model_abs_dir: The directory that saves summary, checkpoints, config etc.
        device: Training in CPU or GPU
        ma_name: Multi-agent name
        train_mode: Is training or inference
        last_ckpt: The checkpoint to restore

        nn_config: nn model config

        nn: nn # Neural network models file
        seed: null # Random seed
        write_summary_per_step: 1000 # Write summaries in TensorBoard every N steps
        save_model_per_step: 5000 # Save model every N steps

        use_replay_buffer: true # Whether using prioritized replay buffer
        use_priority: true # Whether using PER importance ratio

        ensemble_q_num: 2 # Number of Qs
        ensemble_q_sample: 2 # Number of min Qs

        burn_in_step: 0 # Burn-in steps in R2D2
        n_step: 1 # Update Q function by N-steps
        seq_encoder: null # RNN | ATTN

        batch_size: 256 # Batch size for training

        tau: 0.005 # Coefficient of updating target network
        update_target_per_step: 1 # Update target network every N steps

        init_log_alpha: -2.3 # The initial log_alpha
        use_auto_alpha: true # Whether using automating entropy adjustment
        target_d_alpha: 0.98 # Target discrete alpha ratio
        target_c_alpha: 1.0 # Target continuous alpha ratio
        d_policy_entropy_penalty: 0.5 # Discrete policy entropy penalty ratio

        learning_rate: 0.0003 # Learning rate of all optimizers

        gamma: 0.99 # Discount factor
        v_lambda: 1.0 # Discount factor for V-trace
        v_rho: 1.0 # Rho for V-trace
        v_c: 1.0 # C for V-trace
        clip_epsilon: 0.2 # Epsilon for q clip

        discrete_dqn_like: false # Whether using policy or only Q network if discrete is in action spaces
        use_n_step_is: true # Whether using importance sampling
        siamese: null # ATC | BYOL
        siamese_use_q: false # Whether using contrastive q
        siamese_use_adaptive: false # Whether using adaptive weights
        use_prediction: false # Whether training a transition model
        transition_kl: 0.8 # The coefficient of KL of transition and standard normal
        use_extra_data: true # Whether using extra data to train prediction model
        curiosity: null # FORWARD | INVERSE
        curiosity_strength: 1 # Curiosity strength if using curiosity
        use_rnd: false # Whether using RND
        rnd_n_sample: 10 # RND sample times
        use_normalization: false # Whether using observation normalization
        action_noise: null # [noise_min, noise_max]
        """
        self._kwargs = locals()

        self.obs_names = obs_names
        self.obs_shapes = obs_shapes
        self.d_action_sizes = d_action_sizes
        self.d_action_summed_size = sum(d_action_sizes)
        self.d_action_branch_size = len(d_action_sizes)
        self.c_action_size = c_action_size
        self.model_abs_dir = model_abs_dir
        self.ma_name = ma_name
        self.train_mode = train_mode

        self.use_replay_buffer = use_replay_buffer
        self.use_priority = use_priority

        self.ensemble_q_num = ensemble_q_num
        self.ensemble_q_sample = ensemble_q_sample

        self.burn_in_step = burn_in_step
        self.n_step = n_step
        self.seq_encoder = seq_encoder

        self.write_summary_per_step = int(write_summary_per_step)
        self.save_model_per_step = int(save_model_per_step)
        self.batch_size = batch_size
        self.tau = tau
        self.update_target_per_step = update_target_per_step
        self.use_auto_alpha = use_auto_alpha
        self.target_d_alpha = target_d_alpha
        self.target_c_alpha = target_c_alpha
        self.d_policy_entropy_penalty = d_policy_entropy_penalty
        self.gamma = gamma
        self.v_lambda = v_lambda
        self.v_rho = v_rho
        self.v_c = v_c
        self.clip_epsilon = clip_epsilon

        self.discrete_dqn_like = discrete_dqn_like
        self.use_n_step_is = use_n_step_is
        self.siamese = siamese
        self.siamese_use_q = siamese_use_q
        self.siamese_use_adaptive = siamese_use_adaptive
        self.use_prediction = use_prediction
        self.transition_kl = transition_kl
        self.use_extra_data = use_extra_data
        self.curiosity = curiosity
        self.curiosity_strength = curiosity_strength
        self.use_rnd = use_rnd
        self.rnd_n_sample = rnd_n_sample
        self.use_normalization = use_normalization
        self.action_noise = action_noise
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.use_pr = False  # 是否使用策略正则化 True/False
        if self.use_pr:
            self.llm_wrapper = LLMWrapper(call_type="pr", device=self.device)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.summary_writer = None
        if summary_path and self.model_abs_dir and self.train_mode:
            summary_path = Path(self.model_abs_dir).joinpath(summary_path)
            self.summary_writer = SummaryWriter(str(summary_path))
            self.summary_available = True

        self._set_logger()

        self._profiler = UnifiedElapsedTimer(self._logger)

        self._build_model(nn, nn_config, init_log_alpha, learning_rate)
        self._build_ckpt()
        self._init_replay_buffer(replay_config)
        self._init_or_restore(int(last_ckpt) if last_ckpt is not None else None)

    def _set_logger(self):
        if self.ma_name is None:
            self._logger = logging.getLogger('sac.base')
        else:
            self._logger = logging.getLogger(f'sac.base.{self.ma_name}')

    def _build_model(self, nn, nn_config: Optional[dict], init_log_alpha: float, learning_rate: float) -> None:
        """
        Initialize variables, network models and optimizers 用于构建模型、初始化网络和优化器
        """
        if nn_config is None:
            nn_config = {}
        nn_config = defaultdict(dict, nn_config)
        if nn_config['rep'] is None:
            nn_config['rep'] = {}
        if nn_config['policy'] is None:
            nn_config['policy'] = {}

        self.global_step = torch.tensor(0, dtype=torch.int64, requires_grad=False, device='cpu')

        self._gamma_ratio = torch.logspace(0, self.n_step - 1, self.n_step, self.gamma, device=self.device)
        self._lambda_ratio = torch.logspace(0, self.n_step - 1, self.n_step, self.v_lambda, device=self.device)

        self.v_rho = torch.tensor(self.v_rho, device=self.device)
        self.v_c = torch.tensor(self.v_c, device=self.device)

        d_action_list = [np.eye(d_action_size, dtype=np.float32)[0]
                         for d_action_size in self.d_action_sizes]
        self._padding_action = np.concatenate(d_action_list + [np.zeros(self.c_action_size, dtype=np.float32)], axis=-1)

        def adam_optimizer(params):
            return optim.Adam(params, lr=learning_rate)

        """ NORMALIZATION """
        if self.use_normalization:
            self.normalizer_step = torch.tensor(0, dtype=torch.int32, device=self.device, requires_grad=False)
            self.running_means = []
            self.running_variances = []
            for shape in self.obs_shapes:
                self.running_means.append(torch.zeros(shape, device=self.device))
                self.running_variances.append(torch.ones(shape, device=self.device))

            p_self = self

            class ModelRep(nn.ModelRep):
                def forward(self, obs_list, *args, **kwargs):
                    obs_list = [
                        torch.clamp(
                            (obs - mean) / torch.sqrt(variance / (p_self.normalizer_step + 1)),
                            -5, 5
                        ) for obs, mean, variance in zip(obs_list,
                                                         p_self.running_means,
                                                         p_self.running_variances)
                    ]

                    return super().forward(obs_list, *args, **kwargs)
        else:
            ModelRep = nn.ModelRep

        """ REPRESENTATION """
        if self.seq_encoder == SEQ_ENCODER.RNN:
            self.model_rep: ModelBaseRNNRep = ModelRep(self.obs_shapes,
                                                       self.d_action_sizes, self.c_action_size,
                                                       False, self.train_mode,
                                                       self.model_abs_dir,
                                                       **nn_config['rep']).to(self.device)
            self.model_target_rep: ModelBaseRNNRep = ModelRep(self.obs_shapes,
                                                              self.d_action_sizes, self.c_action_size,
                                                              True, self.train_mode,
                                                              self.model_abs_dir,
                                                              **nn_config['rep']).to(self.device)
            # Get represented state and seq_hidden_state_shape
            test_obs_list = [torch.rand(self.batch_size, 1, *obs_shape, device=self.device) for obs_shape in self.obs_shapes]
            test_pre_action = torch.rand(self.batch_size, 1, self.d_action_summed_size + self.c_action_size, device=self.device)
            test_state, test_rnn_state = self.model_rep(test_obs_list,
                                                        test_pre_action)
            state_size, self.seq_hidden_state_shape = test_state.shape[-1], test_rnn_state.shape[1:]

        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            self.model_rep: ModelBaseAttentionRep = ModelRep(self.obs_shapes,
                                                             self.d_action_sizes, self.c_action_size,
                                                             False, self.train_mode,
                                                             self.model_abs_dir,
                                                             **nn_config['rep']).to(self.device)
            self.model_target_rep: ModelBaseAttentionRep = ModelRep(self.obs_shapes,
                                                                    self.d_action_sizes, self.c_action_size,
                                                                    True, self.train_mode,
                                                                    self.model_abs_dir,
                                                                    **nn_config['rep']).to(self.device)
            # Get represented state and seq_hidden_state_shape
            test_index = torch.zeros((self.batch_size, 1), dtype=torch.int32, device=self.device)
            test_obs_list = [torch.rand(self.batch_size, 1, *obs_shape, device=self.device) for obs_shape in self.obs_shapes]
            test_pre_action = torch.rand(self.batch_size, 1, self.d_action_summed_size + self.c_action_size, device=self.device)
            test_state, test_attn_state, _ = self.model_rep(test_index,
                                                            test_obs_list,
                                                            test_pre_action)
            state_size, self.seq_hidden_state_shape = test_state.shape[-1], test_attn_state.shape[2:]

        else:
            self.model_rep: ModelBaseSimpleRep = ModelRep(self.obs_shapes,
                                                          False, self.train_mode,
                                                          self.model_abs_dir,
                                                          **nn_config['rep']).to(self.device)
            self.model_target_rep: ModelBaseSimpleRep = ModelRep(self.obs_shapes,
                                                                 True, self.train_mode,
                                                                 self.model_abs_dir,
                                                                 **nn_config['rep']).to(self.device)
            # Get represented state dimension
            test_obs_list = [torch.rand(self.batch_size, *obs_shape, device=self.device) for obs_shape in self.obs_shapes]
            test_state = self.model_rep(test_obs_list)
            state_size = test_state.shape[-1]

        for param in self.model_target_rep.parameters():
            param.requires_grad = False

        self.state_size = state_size
        self._logger.info(f'State size: {state_size}')

        if len(list(self.model_rep.parameters())) > 0:
            self.optimizer_rep = adam_optimizer(self.model_rep.parameters())
        else:
            self.optimizer_rep = None

        """ Q """
        self.model_q_list: List[ModelBaseQ] = [nn.ModelQ(state_size,
                                                         self.d_action_sizes,
                                                         self.c_action_size,
                                                         False,
                                                         self.train_mode,
                                                         self.model_abs_dir).to(self.device)
                                               for _ in range(self.ensemble_q_num)]  # 2个q网络

        self.model_target_q_list: List[ModelBaseQ] = [nn.ModelQ(state_size,
                                                                self.d_action_sizes,
                                                                self.c_action_size,
                                                                True,
                                                                self.train_mode,
                                                                self.model_abs_dir).to(self.device)
                                                      for _ in range(self.ensemble_q_num)]  # 2个target_q网络
        for model_target_q in self.model_target_q_list:
            for param in model_target_q.parameters():
                param.requires_grad = False

        self.optimizer_q_list = [adam_optimizer(self.model_q_list[i].parameters()) for i in range(self.ensemble_q_num)]

        """ POLICY """
        self.model_policy: ModelBasePolicy = nn.ModelPolicy(state_size, self.d_action_sizes, self.c_action_size,
                                                            self.train_mode,
                                                            self.model_abs_dir,
                                                            **nn_config['policy']).to(self.device)  # 1个actor网络
        self.optimizer_policy = adam_optimizer(self.model_policy.parameters())

        """ SIAMESE REPRESENTATION LEARNING 孪生网络表示学习"""
        if self.siamese in (SIAMESE.ATC, SIAMESE.BYOL):
            test_encoder_list = self.model_rep.get_augmented_encoders(test_obs_list)
            if not isinstance(test_encoder_list, tuple):
                test_encoder_list = [test_encoder_list, ]

            if self.siamese == SIAMESE.ATC:
                self.contrastive_weight_list = [torch.randn((test_encoder.shape[-1], test_encoder.shape[-1]),
                                                            requires_grad=True,
                                                            device=self.device) for test_encoder in test_encoder_list]
                self.optimizer_siamese = adam_optimizer(self.contrastive_weight_list)

            elif self.siamese == SIAMESE.BYOL:
                self.model_rep_projection_list: List[ModelBaseRepProjection] = [
                    nn.ModelRepProjection(test_encoder.shape[-1]).to(self.device) for test_encoder in test_encoder_list]
                self.model_target_rep_projection_list: List[ModelBaseRepProjection] = [
                    nn.ModelRepProjection(test_encoder.shape[-1]).to(self.device) for test_encoder in test_encoder_list]

                test_projection_list = [pro(test_encoder) for pro, test_encoder in zip(self.model_rep_projection_list, test_encoder_list)]
                self.model_rep_prediction_list: List[ModelBaseRepPrediction] = [
                    nn.ModelRepPrediction(test_projection.shape[-1]).to(self.device) for test_projection in test_projection_list]
                self.optimizer_siamese = adam_optimizer(chain(*[pro.parameters() for pro in self.model_rep_projection_list],
                                                              *[pre.parameters() for pre in self.model_rep_prediction_list]))

        """ RECURRENT PREDICTION MODELS 递归预测模型"""
        if self.use_prediction:
            self.model_transition: ModelBaseTransition = nn.ModelTransition(state_size,
                                                                            self.d_action_summed_size,
                                                                            self.c_action_size,
                                                                            self.use_extra_data).to(self.device)
            self.model_reward: ModelBaseReward = nn.ModelReward(state_size).to(self.device)
            self.model_observation: ModelBaseObservation = nn.ModelObservation(state_size, self.obs_shapes,
                                                                               self.use_extra_data).to(self.device)

            self.optimizer_prediction = adam_optimizer(chain(self.model_transition.parameters(),
                                                             self.model_reward.parameters(),
                                                             self.model_observation.parameters()))

        """ ALPHA 温度参数alpha。初始化与温度参数相关的变量和优化器"""
        self.log_d_alpha = torch.tensor(init_log_alpha, dtype=torch.float32, requires_grad=True, device=self.device)
        self.log_c_alpha = torch.tensor(init_log_alpha, dtype=torch.float32, requires_grad=True, device=self.device)

        if self.d_action_sizes:
            d_action_sizes = torch.tensor(self.d_action_sizes, device=self.device)
            d_action_sizes = torch.repeat_interleave(d_action_sizes.type(torch.float32), d_action_sizes)
            self.target_d_alpha = self.target_d_alpha * (-torch.log(1 / d_action_sizes))

        if self.c_action_size:
            self.target_c_alpha = self.target_c_alpha * torch.tensor(-self.c_action_size,
                                                                     dtype=torch.float32,
                                                                     device=self.device)

        if self.use_auto_alpha:
            self.optimizer_alpha = adam_optimizer([self.log_d_alpha, self.log_c_alpha])

        """ CURIOSITY 好奇心驱动"""
        if self.curiosity == CURIOSITY.FORWARD:
            self.model_forward_dynamic: ModelBaseForwardDynamic = nn.ModelForwardDynamic(state_size,
                                                                                         self.d_action_summed_size + self.c_action_size).to(self.device)
            self.optimizer_curiosity = adam_optimizer(self.model_forward_dynamic.parameters())

        elif self.curiosity == CURIOSITY.INVERSE:
            self.model_inverse_dynamic: ModelBaseInverseDynamic = nn.ModelInverseDynamic(state_size,
                                                                                         self.d_action_summed_size + self.c_action_size).to(self.device)
            self.optimizer_curiosity = adam_optimizer(self.model_inverse_dynamic.parameters())

        """ RANDOM NETWORK DISTILLATION 初始化随机网络蒸馏相关的模型和优化器"""
        if self.use_rnd:
            self.model_rnd: ModelBaseRND = nn.ModelRND(state_size, self.d_action_summed_size, self.c_action_size).to(self.device)
            self.model_target_rnd: ModelBaseRND = nn.ModelRND(state_size, self.d_action_summed_size, self.c_action_size).to(self.device)
            for param in self.model_target_rnd.parameters():
                param.requires_grad = False
            self.optimizer_rnd = adam_optimizer(self.model_rnd.parameters())

    # 构建一个检查点（checkpoint）字典 ckpt_dict，用于保存当前模型的状态信息，包括各种模型参数和优化器的状态。用于模型的保存和恢复
    def _build_ckpt(self) -> None:
        self.ckpt_dict = ckpt_dict = {
            'global_step': self.global_step
        }

        """ NORMALIZATION & REPRESENTATION """
        if self.use_normalization:
            ckpt_dict['normalizer_step'] = self.normalizer_step
            for i, v in enumerate(self.running_means):
                ckpt_dict[f'running_means_{i}'] = v
            for i, v in enumerate(self.running_variances):
                ckpt_dict[f'running_variances_{i}'] = v

        if len(list(self.model_rep.parameters())) > 0:
            ckpt_dict['model_rep'] = self.model_rep
            ckpt_dict['model_target_rep'] = self.model_target_rep
            ckpt_dict['optimizer_rep'] = self.optimizer_rep

        """ Q """
        for i in range(self.ensemble_q_num):
            ckpt_dict[f'model_q_{i}'] = self.model_q_list[i]
            ckpt_dict[f'model_target_q_{i}'] = self.model_target_q_list[i]
            ckpt_dict[f'optimizer_q_{i}'] = self.optimizer_q_list[i]

        """ POLICY """
        ckpt_dict['model_policy'] = self.model_policy
        ckpt_dict['optimizer_policy'] = self.optimizer_policy

        """ SIAMESE REPRESENTATION LEARNING """
        if self.siamese == SIAMESE.ATC:
            for i, weight in enumerate(self.contrastive_weight_list):
                ckpt_dict[f'contrastive_weights_{i}'] = weight
        elif self.siamese == SIAMESE.BYOL:
            for i, model_rep_projection in enumerate(self.model_rep_projection_list):
                ckpt_dict[f'model_rep_projection_{i}'] = model_rep_projection
            for i, model_target_rep_projection in enumerate(self.model_target_rep_projection_list):
                ckpt_dict[f'model_target_rep_projection_{i}'] = model_target_rep_projection
            for i, model_rep_prediction in enumerate(self.model_rep_prediction_list):
                ckpt_dict[f'model_rep_prediction_{i}'] = model_rep_prediction
            ckpt_dict['optimizer_siamese'] = self.optimizer_siamese

        """ RECURRENT PREDICTION MODELS """
        if self.use_prediction:
            ckpt_dict['model_transition'] = self.model_transition
            ckpt_dict['model_reward'] = self.model_reward
            ckpt_dict['model_observation'] = self.model_observation
            ckpt_dict['optimizer_prediction'] = self.optimizer_prediction

        """ ALPHA """
        ckpt_dict['log_d_alpha'] = self.log_d_alpha
        ckpt_dict['log_c_alpha'] = self.log_c_alpha
        if self.use_auto_alpha:
            ckpt_dict['optimizer_alpha'] = self.optimizer_alpha

        """ CURIOSITY """
        if self.curiosity is not None:
            if self.curiosity == CURIOSITY.FORWARD:
                ckpt_dict['model_forward_dynamic'] = self.model_forward_dynamic
            elif self.curiosity == CURIOSITY.INVERSE:
                ckpt_dict['model_inverse_dynamic'] = self.model_inverse_dynamic
            ckpt_dict['optimizer_curiosity'] = self.optimizer_curiosity

        """ RANDOM NETWORK DISTILLATION """
        if self.use_rnd:
            ckpt_dict['model_rnd'] = self.model_rnd
            ckpt_dict['optimizer_rnd'] = self.optimizer_rnd

    def _init_or_restore(self, last_ckpt: int) -> None:
        """
        Initialize network weights from scratch or restore from model_abs_dir
        """
        self.ckpt_dir = None
        if self.model_abs_dir:
            self.ckpt_dir = ckpt_dir = self.model_abs_dir.joinpath('model')

            ckpts = []
            if ckpt_dir.exists():
                for ckpt_path in ckpt_dir.glob('*.pth'):
                    ckpts.append(int(ckpt_path.stem))
                ckpts.sort()
            else:
                ckpt_dir.mkdir()

            if ckpts:
                if last_ckpt is None:
                    last_ckpt = ckpts[-1]
                else:
                    assert last_ckpt in ckpts

                ckpt_restore_path = ckpt_dir.joinpath(f'{last_ckpt}.pth')
                ckpt_restore = torch.load(ckpt_restore_path, map_location=self.device)
                for name, model in self.ckpt_dict.items():
                    if name not in ckpt_restore:
                        self._logger.warning(f'{name} not in {last_ckpt}.pth')
                        continue

                    if isinstance(model, torch.Tensor):
                        model.data = ckpt_restore[name]
                    else:
                        try:
                            model.load_state_dict(ckpt_restore[name])
                        except RuntimeError as e:
                            self._logger.error(e)
                        if isinstance(model, nn.Module):
                            if self.train_mode:
                                model.train()
                            else:
                                model.eval()

                self.global_step = self.global_step.to('cpu')

                self._logger.info(f'Restored from {ckpt_restore_path}')

                if self.train_mode and self.use_replay_buffer:
                    self.replay_buffer.load(ckpt_dir, last_ckpt)

                    self._logger.info(f'Replay buffer restored')
            else:
                self._logger.info('Initializing from scratch')
                self._update_target_variables()

    def _init_replay_buffer(self, replay_config: Optional[dict] = None) -> None:
        if self.train_mode:
            if self.use_replay_buffer:
                replay_config = {} if replay_config is None else replay_config
                self.replay_buffer = PrioritizedReplayBuffer(batch_size=self.batch_size, **replay_config)  # 创建一个优先回放缓冲区
            else:
                self.batch_buffer = BatchBuffer(self.burn_in_step,
                                                self.n_step,
                                                self._padding_action,
                                                self.batch_size)

    def set_train_mode(self, train_mode=True):
        self.train_mode = train_mode

    def save_model(self, save_replay_buffer=False) -> None:
        if self.ckpt_dir:
            global_step = self.get_global_step()
            ckpt_path = self.ckpt_dir.joinpath(f'{global_step}.pth')

            torch.save({
                k: v if isinstance(v, torch.Tensor) else v.state_dict()
                for k, v in self.ckpt_dict.items()
            }, ckpt_path)
            self._logger.info(f"Model saved at {ckpt_path}")

            if self.use_replay_buffer and save_replay_buffer:
                self.replay_buffer.save(self.ckpt_dir, global_step)

    def write_constant_summaries(self, constant_summaries, iteration=None) -> None:
        """
        Write constant information from sac_main.py, such as reward, iteration, etc.
        """
        if self.summary_writer is not None:
            for s in constant_summaries:
                self.summary_writer.add_scalar(s['tag'], s['simple_value'],
                                               self.get_global_step() if iteration is None else iteration)

        self.summary_writer.flush()

    def _increase_global_step(self) -> int:
        self.global_step.add_(1)

        return self.global_step.item()

    def get_global_step(self) -> int:
        return self.global_step.item()

    def get_initial_action(self, batch_size) -> np.ndarray:
        if self.d_action_sizes:
            d_actions = [np.random.randint(0, d_action_size, size=batch_size)
                         for d_action_size in self.d_action_sizes]
            d_actions = [np.eye(d_action_size, dtype=np.int32)[d_action]
                         for d_action, d_action_size in zip(d_actions, self.d_action_sizes)]
            d_action = np.concatenate(d_actions, axis=-1).astype(np.float32)
        else:
            d_action = np.zeros((batch_size, 0), dtype=np.float32)

        c_action = np.zeros([batch_size, self.c_action_size], dtype=np.float32)

        return np.concatenate([d_action, c_action], axis=-1)

    def get_initial_seq_hidden_state(self, batch_size, get_numpy=True) -> Union[np.ndarray, torch.Tensor]:
        assert self.seq_encoder is not None

        if get_numpy:
            return np.zeros([batch_size, *self.seq_hidden_state_shape], dtype=np.float32)
        else:
            return torch.zeros([batch_size, *self.seq_hidden_state_shape], device=self.device)

    @torch.no_grad()
    def _update_target_variables(self, tau=1.) -> None:
        """
        Soft (momentum) update target networks (default hard)
        """
        target = self.model_target_rep.parameters()
        source = self.model_rep.parameters()

        for i in range(self.ensemble_q_num):
            target = chain(target, self.model_target_q_list[i].parameters())
            source = chain(source, self.model_q_list[i].parameters())

        if self.siamese == 'BYOL':
            target = chain(target, *[t_pro.parameters() for t_pro in self.model_target_rep_projection_list])
            source = chain(source, *[pro.parameters() for pro in self.model_rep_projection_list])

        for target_param, param in zip(target, source):
            target_param.data.copy_(
                target_param.data * (1. - tau) + param.data * tau
            )

    @torch.no_grad()
    def _udpate_normalizer(self, obs_list: List[torch.Tensor]) -> None:
        self.normalizer_step.add_(obs_list[0].shape[0])

        input_to_old_means = [obs_list[i] - self.running_means[i] for i in range(len(obs_list))]
        new_means = [self.running_means[i] + torch.sum(
            input_to_old_means[i] / self.normalizer_step, dim=0
        ) for i in range(len(obs_list))]

        input_to_new_means = [obs_list[i] - new_means[i] for i in range(len(obs_list))]
        new_variances = [self.running_variances[i] + torch.sum(
            input_to_new_means[i] * input_to_old_means[i], dim=0
        ) for i in range(len(obs_list))]

        for t_p, p in zip(self.running_means + self.running_variances, new_means + new_variances):
            t_p.copy_(p)

    #################### ! GET ACTION ####################

    @torch.no_grad()
    def rnd_sample_d_action(self, state: torch.Tensor,
                            d_policy: distributions.Categorical) -> torch.Tensor:
        """
        Sample action `self.rnd_n_sample` times, 
        choose the action that has the max (model_rnd(s, a) - model_target_rnd(s, a))**2

        Args:
            state: [batch, state_size]
            d_policy: [batch, d_action_summed_size]

        Returns:
            d_action: [batch, d_action_summed_size]
        """
        batch = state.shape[0]
        n_sample = self.rnd_n_sample

        d_actions = d_policy.sample((n_sample,))  # [n_sample, batch, d_action_summed_size]
        d_actions = d_actions.transpose(0, 1)  # [batch, n_sample, d_action_summed_size]

        d_rnd = self.model_rnd.cal_d_rnd(state)  # [batch, d_action_summed_size, f]
        t_d_rnd = self.model_target_rnd.cal_d_rnd(state)  # [batch, d_action_summed_size, f]
        d_rnd = torch.repeat_interleave(torch.unsqueeze(d_rnd, 1), n_sample, dim=1)  # [batch, n_sample, d_action_summed_size, f]
        t_d_rnd = torch.repeat_interleave(torch.unsqueeze(t_d_rnd, 1), n_sample, dim=1)  # [batch, n_sample, d_action_summed_size, f]

        _i = d_actions.unsqueeze(-1)  # [batch, n_sample, d_action_summed_size, 1]
        d_rnd = (torch.repeat_interleave(_i, d_rnd.shape[-1], dim=-1) * d_rnd).sum(-2)
        # [batch, n_sample, d_action_summed_size, f] -> [batch, n_sample, f]
        t_d_rnd = (torch.repeat_interleave(_i, t_d_rnd.shape[-1], dim=-1) * t_d_rnd).sum(-2)
        # [batch, n_sample, d_action_summed_size, f] -> [batch, n_sample, f]

        d_loss = torch.sum(torch.pow(d_rnd - t_d_rnd, 2), dim=-1)  # [batch, n_sample]
        d_idx = torch.argmax(d_loss, dim=1)  # [batch, ]

        return d_actions[torch.arange(batch), d_idx]

    @torch.no_grad()
    def rnd_sample_c_action(self, state: torch.Tensor,
                            c_policy: distributions.Normal) -> torch.Tensor:
        """
        Sample action `self.rnd_n_sample` times, 
        choose the action that has the max (model_rnd(s, a) - model_target_rnd(s, a))**2

        Args:
            state: [batch, state_size]
            c_policy: [batch, c_action_size]

        Returns:
            c_action: [batch, c_action_size]
        """
        batch = state.shape[0]
        n_sample = self.rnd_n_sample

        c_actions = torch.tanh(c_policy.sample((n_sample,)))  # [n_sample, batch, c_action_size]
        c_actions = c_actions.transpose(0, 1)  # [batch, n_sample, c_action_size]

        states = torch.repeat_interleave(torch.unsqueeze(state, 1), n_sample, dim=1)
        # [batch, state_size] -> [batch, 1, state_size] -> [batch, n_sample, state_size]
        c_rnd = self.model_rnd.cal_c_rnd(states, c_actions)  # [batch, n_sample, f]
        t_c_rnd = self.model_target_rnd.cal_c_rnd(states, c_actions)  # [batch, n_sample, f]

        c_loss = torch.sum(torch.pow(c_rnd - t_c_rnd, 2), dim=-1)  # [batch, n_sample]
        c_idx = torch.argmax(c_loss, dim=1)  # [batch, ]

        return c_actions[torch.arange(batch), c_idx]

    @torch.no_grad()
    def _random_action(self, d_action, c_action) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.action_noise is None:
            return d_action, c_action

        batch = max(d_action.shape[0], c_action.shape[0])

        action_noise = torch.linspace(*self.action_noise, steps=batch, device=self.device)  # [batch, ]

        if self.d_action_sizes:
            random_d_actions = [torch.argmax(torch.rand(batch, d_action_size, device=self.device), dim=-1)
                                for d_action_size in self.d_action_sizes]
            random_d_actions = [functional.one_hot(random_d_action, d_action_size)
                                for random_d_action, d_action_size in zip(random_d_actions, self.d_action_sizes)]
            random_d_action = torch.concat(random_d_actions, axis=-1).type(torch.float32)

            mask = torch.rand(batch, device=self.device) < action_noise
            d_action[mask] = random_d_action[mask]

        if self.c_action_size:
            c_action = torch.tanh(torch.atanh(c_action) + torch.randn(batch, self.c_action_size, device=self.device) * action_noise.unsqueeze(1))

        return d_action, c_action

    @torch.no_grad()
    def _choose_action(self,
                       obs_list: List[torch.Tensor],
                       state: torch.Tensor,
                       disable_sample: bool = False,
                       force_rnd_if_available: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: [batch, state_size]

        Returns:
            action: [batch, action_size]
            prob: [batch, action_size]
        """
        batch = state.shape[0]
        d_policy, c_policy = self.model_policy(state, obs_list)

        if self.d_action_sizes:
            if self.discrete_dqn_like:
                d_qs, _ = self.model_q_list[0](state, c_policy.sample() if self.c_action_size else None)
                d_action_list = [torch.argmax(d_q, dim=-1) for d_q in d_qs.split(self.d_action_sizes, dim=-1)]
                d_action_list = [functional.one_hot(d_action, d_action_size).type(torch.float32)
                                 for d_action, d_action_size in zip(d_action_list, self.d_action_sizes)]
                d_action = torch.concat(d_action_list, dim=-1)

                if self.train_mode:
                    mask = torch.rand(batch) < 0.2

                    d_dist_list = [distributions.OneHotCategorical(logits=torch.ones((batch, d_action_size),
                                                                                     device=self.device))
                                   for d_action_size in self.d_action_sizes]
                    random_d_action = torch.concat([dist.sample() for dist in d_dist_list], dim=-1)

                    d_action[mask] = random_d_action[mask]
            else:
                if disable_sample:
                    d_action = d_policy.sample_deter()
                elif self.use_rnd and (self.train_mode or force_rnd_if_available):
                    d_action = self.rnd_sample_d_action(state, d_policy)
                else:
                    d_action = d_policy.sample()
        else:
            d_action = torch.zeros(0, device=self.device)

        if self.c_action_size:
            if disable_sample:
                c_action = torch.tanh(c_policy.mean)
            elif self.use_rnd and (self.train_mode or force_rnd_if_available):
                c_action = self.rnd_sample_c_action(state, c_policy)
            else:
                c_action = torch.tanh(c_policy.sample())
        else:
            c_action = torch.zeros(0, device=self.device)

        d_action, c_action = self._random_action(d_action, c_action)

        prob = torch.ones((batch,
                           self.d_action_summed_size + self.c_action_size),
                          device=self.device)  # [batch, action_size]
        if self.d_action_sizes and not self.discrete_dqn_like:
            prob[:, :self.d_action_summed_size] = d_policy.probs  # [batch, d_action_summed_size]
        if self.c_action_size:
            c_prob = squash_correction_prob(c_policy, torch.atanh(c_action))  # [batch, c_action_size]
            prob[:, -self.c_action_size:] = c_prob  # [batch, c_action_size]

        return torch.cat([d_action, c_action], dim=-1), prob

    @torch.no_grad()
    def choose_action(self,
                      obs_list: List[np.ndarray],
                      disable_sample: bool = False,
                      force_rnd_if_available: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            obs_list (np): list([batch, *obs_shapes_i], ...)

        Returns:
            action (np): [batch, action_size]
            prob (np): [batch, action_size]
        """
        obs_list = [torch.from_numpy(obs).to(self.device) for obs in obs_list]
        state = self.model_rep(obs_list)

        action, prob = self._choose_action(obs_list, state, disable_sample, force_rnd_if_available)
        return action.detach().cpu().numpy(), prob.detach().cpu().numpy()

    @torch.no_grad()
    def choose_rnn_action(self,
                          obs_list: List[np.ndarray],
                          pre_action: np.ndarray,
                          rnn_state: np.ndarray,
                          disable_sample: bool = False,
                          force_rnd_if_available: bool = False) -> Tuple[np.ndarray,
                                                                         np.ndarray,
                                                                         np.ndarray]:
        """
        Args:
            obs_list (np): list([batch, *obs_shapes_i], ...)
            pre_action (np): [batch, action_size]
            rnn_state (np): [batch, *seq_hidden_state_shape]
        Returns:
            action (np): [batch, action_size]
            prob (np): [batch, action_size]
            rnn_state (np): [batch, *seq_hidden_state_shape]
        """
        obs_list = [torch.from_numpy(obs).to(self.device) for obs in obs_list]
        pre_action = torch.from_numpy(pre_action).to(self.device)
        rnn_state = torch.from_numpy(rnn_state).to(self.device)

        obs_list = [obs.unsqueeze(1) for obs in obs_list]
        pre_action = pre_action.unsqueeze(1)
        state, next_rnn_state = self.model_rep(obs_list, pre_action, rnn_state)
        # state: [batch, 1, state_size]
        state = state.squeeze(1)
        obs_list = [obs.squeeze(1) for obs in obs_list]

        action, prob = self._choose_action(obs_list, state, disable_sample, force_rnd_if_available)

        return (action.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                next_rnn_state.detach().cpu().numpy())

    @torch.no_grad()
    def choose_attn_action(self,
                           ep_indexes: np.ndarray,
                           ep_padding_masks: np.ndarray,
                           ep_obses_list: List[np.ndarray],
                           ep_pre_actions: np.ndarray,
                           ep_attn_states: np.ndarray,

                           disable_sample: bool = False,
                           force_rnd_if_available: bool = False) -> Tuple[np.ndarray,
                                                                          np.ndarray,
                                                                          np.ndarray]:
        """
        Args:
            ep_indexes (np.int32): [batch, ep_len]
            ep_padding_masks (bool): [batch, ep_len]
            ep_obses_list (np): list([batch, ep_len, *obs_shapes_i], ...)
            ep_pre_actions (np): [batch, ep_len, action_size]
            ep_attn_states (np): [batch, ep_len, *seq_hidden_state_shape]

        Returns:
            action (np): [batch, action_size]
            prob (np): [batch, action_size]
            attn_state (np): [batch, *attn_state_shape]
        """
        ep_indexes = torch.from_numpy(ep_indexes).to(self.device)
        ep_padding_masks = torch.from_numpy(ep_padding_masks).to(self.device)
        ep_obses_list = [torch.from_numpy(obs).to(self.device) for obs in ep_obses_list]
        ep_pre_actions = torch.from_numpy(ep_pre_actions).to(self.device)
        ep_attn_states = torch.from_numpy(ep_attn_states).to(self.device)

        state, next_attn_state, _ = self.model_rep(ep_indexes,
                                                   ep_obses_list, ep_pre_actions,
                                                   query_length=1,
                                                   hidden_state=ep_attn_states,
                                                   is_prev_hidden_state=False,
                                                   padding_mask=ep_padding_masks)
        # state: [batch, 1, state_size]
        # next_attn_state: [batch, 1, *attn_state_shape]

        state = state.squeeze(1)
        next_attn_state = next_attn_state.squeeze(1)

        action, prob = self._choose_action([ep_obses[:, -1] for ep_obses in ep_obses_list],
                                           state,
                                           disable_sample,
                                           force_rnd_if_available)

        return (action.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                next_attn_state.detach().cpu().numpy())

    #################### ! GET STATES ####################

    def get_m_data(self,
                   bn_indexes: torch.Tensor,
                   bn_padding_masks: torch.Tensor,
                   bn_obses_list: List[torch.Tensor],
                   bn_actions: torch.Tensor,
                   next_obs_list: torch.Tensor) -> Tuple[torch.Tensor,
                                                         torch.Tensor,
                                                         List[torch.Tensor],
                                                         torch.Tensor]:
        """
        Args:
            bn_indexes (torch.int32): [batch, b + n]
            bn_padding_masks (torch.bool): [batch, b + n]
            bn_obses_list: list([batch, b + n, *obs_shapes_i], ...)
            bn_actions: [batch, b + n, action_size]
            next_obs_list: list([batch, *obs_shapes_i], ...)

        Returns:
            m_indexes: [batch, b + n + 1]
            m_padding_masks: [batch, b + n + 1]
            m_obses_list: list([batch, b + n + 1, *obs_shapes_i], ...)
            m_pre_actions: [batch, b + n + 1, action_size]
        """

        m_indexes = torch.concat([bn_indexes, bn_indexes[:, -1:] + 1], dim=1)
        m_padding_masks = torch.concat([bn_padding_masks,
                                        torch.zeros_like(bn_padding_masks[:, -1:], dtype=torch.bool)], dim=1)
        m_obses_list = [torch.cat([bn_obses, next_obs.unsqueeze(1)], dim=1)
                        for bn_obses, next_obs in zip(bn_obses_list, next_obs_list)]
        m_pre_actions = gen_pre_n_actions(bn_actions, keep_last_action=True)

        return m_indexes, m_padding_masks, m_obses_list, m_pre_actions

    def get_l_states(self,
                     l_indexes: torch.Tensor,
                     l_padding_masks: torch.Tensor,
                     l_obses_list: List[torch.Tensor],
                     l_pre_actions: Optional[torch.Tensor] = None,
                     f_seq_hidden_states: Optional[torch.Tensor] = None,
                     is_target=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            l_indexes: [batch, l]
            l_padding_masks: [batch, l]
            l_obses_list: list([batch, l, *obs_shapes_i], ...)
            l_pre_actions: [batch, l, action_size]
            f_seq_hidden_states: [batch, 1, *seq_hidden_state_shape]

        Returns:
            l_states: [batch, l, state_size]
            next_f_rnn_states (optional): [batch, 1, rnn_state_size]
            next_l_attn_states (optional): [batch, l, attn_state_size]
        """

        model_rep = self.model_target_rep if is_target else self.model_rep

        if self.seq_encoder is None:
            l_states = model_rep(l_obses_list)

            return l_states, None  # [batch, l, state_size]

        elif self.seq_encoder == SEQ_ENCODER.RNN:
            l_states, next_rnn_state = model_rep(l_obses_list,
                                                 l_pre_actions,
                                                 f_seq_hidden_states[:, 0],
                                                 padding_mask=l_padding_masks)
            next_f_rnn_states = next_rnn_state.unsqueeze(dim=1)

            return l_states, next_f_rnn_states  # [batch, l, state_size], [batch, 1, rnn_state_size]

        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            l_states, l_attn_states, _ = model_rep(l_indexes,
                                                   l_obses_list,
                                                   l_pre_actions,
                                                   query_length=l_indexes.shape[1],
                                                   hidden_state=f_seq_hidden_states,
                                                   is_prev_hidden_state=True,
                                                   padding_mask=l_padding_masks)

            return l_states, l_attn_states  # [batch, l, state_size], [batch, l, attn_state_size]

    def get_l_states_with_seq_hidden_states(
        self,
        l_indexes: torch.Tensor,
        l_padding_masks: torch.Tensor,
        l_obses_list: List[torch.Tensor],
        l_pre_actions: Optional[torch.Tensor] = None,
        f_seq_hidden_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor,
               Optional[torch.Tensor]]:
        """
        Args:
            l_indexes (torch.int32): [batch, l]
            l_padding_masks (torch.bool): [batch, l]
            l_obses_list: list([batch, l, *obs_shapes_i], ...)
            l_pre_actions: [batch, l, action_size]
            f_seq_hidden_states: [batch, 1, *seq_hidden_state_shape]

        Returns:
            l_states: [batch, l, state_size]
            l_seq_hidden_state: [batch, l, *seq_hidden_state_shape]
        """

        if self.seq_encoder is None:
            l_states = self.model_rep(l_obses_list)

            return l_states, None  # [batch, l, state_size]

        elif self.seq_encoder == SEQ_ENCODER.RNN:
            batch, l, *_ = l_indexes.shape

            l_states = None
            next_l_rnn_states = torch.zeros((batch, l, *f_seq_hidden_states.shape[2:]), device=self.device)

            rnn_state = f_seq_hidden_states[:, 0]
            for t in range(l):
                f_states, rnn_state = self.model_rep([l_obses[:, t:t + 1, ...] for l_obses in l_obses_list],
                                                     l_pre_actions[:, t:t + 1, ...] if l_pre_actions is not None else None,
                                                     rnn_state,
                                                     padding_mask=l_padding_masks[:, t:t + 1])

                if l_states is None:
                    l_states = torch.zeros((batch, l, *f_states.shape[2:]), device=self.device)
                l_states[:, t:t + 1] = f_states

                next_l_rnn_states[:, t] = rnn_state

            return l_states, next_l_rnn_states

        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            return self.get_l_states(l_indexes=l_indexes,
                                     l_padding_masks=l_padding_masks,
                                     l_obses_list=l_obses_list,
                                     l_pre_actions=l_pre_actions,
                                     f_seq_hidden_states=f_seq_hidden_states,
                                     is_target=False)

    @torch.no_grad()
    def get_l_probs(self,
                    l_obses_list: List[torch.Tensor],
                    l_states: torch.Tensor,
                    l_actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            l_obses_list: list([batch, l, *obs_shapes_i], ...)
            l_states: [batch, l, state_size]
            l_actions: [batch, l, action_size]

        Returns:
            l_probs: [batch, l, action_size]
        """

        d_policy, c_policy = self.model_policy(l_states, l_obses_list)

        probs = torch.ones((*l_states.shape[:2], self.d_action_summed_size + self.c_action_size),
                           dtype=torch.float32,
                           device=self.device)  # [batch, l, action_size]

        if self.d_action_sizes:
            probs[..., :self.d_action_summed_size] = d_policy.probs  # [batch, l, d_action_summed_size]

        if self.c_action_size:
            l_selected_c_actions = l_actions[..., -self.c_action_size:]
            c_policy_prob = squash_correction_prob(c_policy, torch.atanh(l_selected_c_actions))
            # [batch, l, c_action_size]
            probs[..., -self.c_action_size:] = c_policy_prob  # [batch, l, c_action_size]

        return probs  # [batch, l, action_size]

    #################### ! COMPUTE LOSS ####################

    @torch.no_grad()
    def get_dqn_like_d_y(self,
                         n_padding_masks: torch.Tensor,
                         n_rewards: torch.Tensor,
                         n_dones: torch.Tensor,
                         stacked_next_n_d_qs: torch.Tensor,
                         stacked_next_target_n_d_qs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            n_padding_masks (torch.bool): [batch, n]
            n_rewards: [batch, n]
            n_dones (torch.bool): [batch, n]
            stacked_next_n_d_qs: [ensemble_q_sample, batch, n, d_action_summed_size]
            stacked_next_target_n_d_qs: [ensemble_q_sample, batch, n, d_action_summed_size]

        Returns:
            y: [batch, 1]
        """
        batch_tensor = torch.arange(n_padding_masks.shape[0], device=self.device)
        last_solid_index = get_last_false_indexes(n_padding_masks, dim=1)  # [batch, ]

        done = n_dones[batch_tensor, last_solid_index].unsqueeze(-1)  # [batch, 1]
        stacked_next_q = stacked_next_n_d_qs[:, batch_tensor, last_solid_index, :]
        # [ensemble_q_sample, batch, d_action_summed_size]
        stacked_next_target_q = stacked_next_target_n_d_qs[:, batch_tensor, last_solid_index, :]
        # [ensemble_q_sample, batch, d_action_summed_size]

        stacked_next_q_list = stacked_next_q.split(self.d_action_sizes, dim=-1)

        mask_stacked_q_list = [functional.one_hot(torch.argmax(stacked_next_q, dim=-1),
                                                  d_action_size)
                               for stacked_next_q, d_action_size in zip(stacked_next_q_list, self.d_action_sizes)]
        mask_stacked_q = torch.concat(mask_stacked_q_list, dim=-1)
        # [ensemble_q_sample, batch, d_action_summed_size]

        stacked_max_next_target_q = torch.sum(stacked_next_target_q * mask_stacked_q,
                                              dim=-1,
                                              keepdim=True)
        # [ensemble_q_sample, batch, 1]
        stacked_max_next_target_q = stacked_max_next_target_q / self.d_action_branch_size

        next_q, _ = torch.min(stacked_max_next_target_q, dim=0)
        # [batch, 1]

        g = torch.sum(self._gamma_ratio * n_rewards, dim=-1, keepdim=True)  # [batch, 1]
        y = g + torch.pow(self.gamma, last_solid_index.unsqueeze(-1) + 1) * next_q * ~done  # [batch, 1]

        return y

    @torch.no_grad()
    def _v_trace(self,
                 n_padding_masks: torch.Tensor,
                 n_rewards: torch.Tensor,
                 n_dones: torch.Tensor,
                 n_mu_probs: torch.Tensor,
                 n_pi_probs: torch.Tensor,
                 n_vs: torch.Tensor,
                 next_n_vs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            n_padding_masks (torch.bool): [batch, ]
            n_rewards: [batch, n]
            n_dones (torch.bool): [batch, n]
            n_mu_probs: [batch, n]
            n_pi_probs: [batch, n]
            n_vs: [batch, n]
            next_n_vs: [batch, n]

        Returns:
            y: [batch, 1]
        """

        td_error = n_rewards + self.gamma * ~n_dones * next_n_vs - n_vs  # [batch, n]
        td_error = self._gamma_ratio * td_error

        if self.use_n_step_is:
            td_error = self._lambda_ratio * td_error

            n_step_is = n_pi_probs / n_mu_probs.clamp(min=1e-8)

            # \rho_t, t \in [s, s+n-1]
            rho = torch.minimum(n_step_is, self.v_rho)  # [batch, n]

            # \prod{c_i}, i \in [s, t-1]
            c = torch.minimum(n_step_is, self.v_c)
            c = torch.cat([torch.ones((n_step_is.shape[0], 1), device=self.device), c[..., :-1]], dim=-1)
            c = torch.cumprod(c, dim=1)

            # \prod{c_i} * \rho_t * td_error
            td_error = c * rho * td_error

        td_error = td_error * ~n_padding_masks
        # \sum{td_error}
        r = torch.sum(td_error, dim=1, keepdim=True)  # [batch, 1]

        # V_s + \sum{td_error}
        y = n_vs[:, 0:1] + r  # [batch, 1]

        return y

    @torch.no_grad()
    def _get_y(self,
               n_padding_masks: torch.Tensor,
               n_obses_list: List[torch.Tensor],
               n_states: torch.Tensor,
               n_actions: torch.Tensor,
               n_rewards: torch.Tensor,
               next_obs_list: List[torch.Tensor],
               next_state: torch.Tensor,
               n_dones: torch.Tensor,
               n_mu_probs: torch.Tensor = None) -> Tuple[Optional[torch.Tensor],
                                                         Optional[torch.Tensor]]:
        """
        Args:
            n_padding_masks (torch.bool): [batch, n]
            n_obses_list: list([batch, n, *obs_shapes_i], ...)
            n_states: [batch, n, state_size]
            n_actions: [batch, n, action_size]
            n_rewards: [batch, n]
            state_: [batch, state_size]
            n_dones (torch.bool): [batch, n]
            n_mu_probs: [batch, n, action_size]

        Returns:
            y: [batch, 1]
        """

        d_alpha = torch.exp(self.log_d_alpha)
        c_alpha = torch.exp(self.log_c_alpha)

        next_n_obses_list = [torch.cat([n_obses[:, 1:, ...], next_obs.unsqueeze(1)], dim=1)
                             for n_obses, next_obs in zip(n_obses_list, next_obs_list)]  # list([batch, n, *obs_shapes_i], ...)
        next_n_states = torch.cat([n_states[:, 1:, ...], next_state.unsqueeze(1)], dim=1)  # [batch, n, state_size]

        d_policy, c_policy = self.model_policy(n_states, n_obses_list)
        next_d_policy, next_c_policy = self.model_policy(next_n_states, next_n_obses_list)

        if self.curiosity is not None:
            if self.curiosity == CURIOSITY.FORWARD:
                approx_next_n_states = self.model_forward_dynamic(n_states, n_actions)  # [batch, n, state_size]
                in_n_rewards = torch.sum(torch.pow(approx_next_n_states - next_n_states, 2), dim=-1) * 0.5  # [batch, n]

            elif self.curiosity == CURIOSITY.INVERSE:
                approx_n_actions = self.model_inverse_dynamic(n_states, next_n_states)  # [batch, n, action_size]
                in_n_rewards = torch.sum(torch.pow(approx_n_actions - n_actions, 2), dim=-1) * 0.5  # [batch, n]

            in_n_rewards = in_n_rewards * self.curiosity_strength  # [batch, n]
            n_rewards += in_n_rewards  # [batch, n]

        if self.c_action_size:
            n_c_actions_sampled = c_policy.rsample()  # [batch, n, c_action_size]
            next_n_c_actions_sampled = next_c_policy.rsample()  # [batch, n, c_action_size]
        else:
            n_c_actions_sampled = torch.zeros(0, device=self.device)
            next_n_c_actions_sampled = torch.zeros(0, device=self.device)

        n_qs_list = [q(n_states, torch.tanh(n_c_actions_sampled)) for q in self.model_target_q_list]
        next_n_qs_list = [q(next_n_states, torch.tanh(next_n_c_actions_sampled)) for q in self.model_target_q_list]
        # ([batch, n, d_action_summed_size], [batch, n, 1])

        n_d_qs_list = [q[0] for q in n_qs_list]  # [batch, n, d_action_summed_size]
        n_c_qs_list = [q[1] for q in n_qs_list]  # [batch, n, 1]

        next_n_d_qs_list = [q[0] for q in next_n_qs_list]  # [batch, n, d_action_summed_size]
        next_n_c_qs_list = [q[1] for q in next_n_qs_list]  # [batch, n, 1]

        d_y, c_y = None, None

        if self.d_action_sizes:
            stacked_next_n_d_qs = torch.stack(next_n_d_qs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, batch, n, d_action_summed_size] -> [ensemble_q_sample, batch, n, d_action_summed_size]

            if self.discrete_dqn_like:
                next_n_d_eval_qs_list = [q(next_n_states, torch.tanh(next_n_c_actions_sampled))[0] for q in self.model_q_list]
                stacked_next_n_d_eval_qs = torch.stack(next_n_d_eval_qs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
                # [ensemble_q_num, batch, n, d_action_summed_size] -> [ensemble_q_sample, batch, n, d_action_summed_size]

                d_y = self.get_dqn_like_d_y(n_padding_masks=n_padding_masks,
                                            n_rewards=n_rewards,
                                            n_dones=n_dones,
                                            stacked_next_n_d_qs=stacked_next_n_d_eval_qs,
                                            stacked_next_target_n_d_qs=stacked_next_n_d_qs)
            else:
                stacked_n_d_qs = torch.stack(n_d_qs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
                # [ensemble_q_num, batch, n, d_action_summed_size] -> [ensemble_q_sample, batch, n, d_action_summed_size]

                mean_n_qs = torch.mean(stacked_n_d_qs, dim=0)  # [batch, n, d_action_summed_size]
                mean_next_n_qs = torch.mean(stacked_next_n_d_qs, dim=0)  # [batch, n, d_action_summed_size]

                probs = d_policy.probs  # [batch, n, d_action_summed_size]
                next_probs = next_d_policy.probs  # [batch, n, d_action_summed_size]
                # ! Note that the probs here is not strict probabilities
                # ! sum(probs) == self.d_action_branch_size
                clipped_probs = probs.clamp(min=1e-8)  # [batch, n, d_action_summed_size]
                clipped_next_probs = next_probs.clamp(min=1e-8)  # [batch, n, d_action_summed_size]
                tmp_n_vs = mean_n_qs - d_alpha * torch.log(clipped_probs)  # [batch, n, d_action_summed_size]
                tmp_next_n_vs = mean_next_n_qs - d_alpha * torch.log(clipped_next_probs)  # [batch, n, d_action_summed_size]

                n_vs = torch.sum(probs * tmp_n_vs, dim=-1) / self.d_action_branch_size  # [batch, n]
                next_n_vs = torch.sum(next_probs * tmp_next_n_vs, dim=-1) / self.d_action_branch_size  # [batch, n]

                if self.use_n_step_is:
                    n_d_actions = n_actions[..., :self.d_action_summed_size]  # [batch, n, d_action_summed_size]
                    n_d_mu_probs = n_mu_probs[..., :self.d_action_summed_size]  # [batch, n, d_action_summed_size]
                    n_d_mu_probs = n_d_mu_probs * n_d_actions  # [batch, n]
                    n_d_mu_probs[n_d_mu_probs == 0.] = 1.
                    n_d_mu_probs = n_d_mu_probs.prod(-1)  # [batch, n]
                    n_d_pi_probs = torch.exp(d_policy.log_prob(n_d_actions).sum(-1))  # [batch, n]

                d_y = self._v_trace(n_padding_masks=n_padding_masks,
                                    n_rewards=n_rewards,
                                    n_dones=n_dones,
                                    n_mu_probs=n_d_mu_probs if self.use_n_step_is else None,
                                    n_pi_probs=n_d_pi_probs if self.use_n_step_is else None,
                                    n_vs=n_vs,
                                    next_n_vs=next_n_vs)

        if self.c_action_size:
            n_actions_log_prob = torch.sum(squash_correction_log_prob(c_policy, n_c_actions_sampled), dim=-1)  # [batch, n]
            next_n_actions_log_prob = torch.sum(squash_correction_log_prob(next_c_policy, next_n_c_actions_sampled), dim=-1)  # [batch, n]

            stacked_n_c_qs = torch.stack(n_c_qs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, batch, n, 1] -> [ensemble_q_sample, batch, n, 1]
            stacked_next_n_c_qs = torch.stack(next_n_c_qs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, batch, n, 1] -> [ensemble_q_sample, batch, n, 1]

            min_n_c_qs, _ = stacked_n_c_qs.min(dim=0)
            min_n_c_qs = min_n_c_qs.squeeze(dim=-1)  # [batch, n]
            min_next_n_c_qs, _ = stacked_next_n_c_qs.min(dim=0)
            min_next_n_c_qs = min_next_n_c_qs.squeeze(dim=-1)  # [batch, n]

            n_vs = min_n_c_qs - c_alpha * n_actions_log_prob  # [batch, n]
            next_n_vs = min_next_n_c_qs - c_alpha * next_n_actions_log_prob  # [batch, n]

            # v = scale_inverse_h(v)
            # next_v = scale_inverse_h(next_v)

            if self.use_n_step_is:
                n_c_actions = n_actions[..., -self.c_action_size:]
                n_c_mu_probs = n_mu_probs[..., -self.c_action_size:]  # [batch, n, c_action_size]
                n_c_pi_probs = squash_correction_prob(c_policy, torch.atanh(n_c_actions))
                # [batch, n, c_action_size]

            c_y = self._v_trace(n_padding_masks=n_padding_masks,
                                n_rewards=n_rewards,
                                n_dones=n_dones,
                                n_mu_probs=n_c_mu_probs.prod(-1) if self.use_n_step_is else None,
                                n_pi_probs=n_c_pi_probs.prod(-1) if self.use_n_step_is else None,
                                n_vs=n_vs,
                                next_n_vs=next_n_vs)

        return d_y, c_y  # [batch, 1]

    def _train_rep_q(self,
                     bn_indexes: List[torch.Tensor],
                     bn_padding_masks: List[torch.Tensor],
                     bn_obses_list: List[torch.Tensor],
                     bn_states: torch.Tensor,
                     bn_target_states: torch.Tensor,
                     bn_actions: torch.Tensor,
                     bn_rewards: torch.Tensor,
                     next_obs_list: List[torch.Tensor],
                     next_state: torch.Tensor,
                     next_target_state: torch.Tensor,
                     bn_dones: torch.Tensor,
                     bn_mu_probs: torch.Tensor,
                     priority_is: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor,
                                                                           Optional[torch.Tensor],
                                                                           Optional[torch.Tensor],
                                                                           Optional[torch.Tensor]]:
        """
        Args:
            bn_indexes (torch.int32): [batch, b + n],
            bn_padding_masks (torch.bool): [batch, b + n],
            bn_obses_list: list([batch, b + n, *obs_shapes_i], ...)
            bn_states: [batch, b + n, state_size]
            bn_target_states: [batch, b + n, state_size]
            bn_actions: [batch, b + n, action_size]
            bn_rewards: [batch, b + n]
            next_obs_list: list([batch, *obs_shapes_i], ...)
            next_state: [batch, state_size]
            next_target_state: [batch, state_size]
            bn_dones (torch.bool): [batch, b + n]
            bn_mu_probs: [batch, b + n, action_size]
            priority_is: [batch, 1]

        Returns:
            loss_q: torch.float32
            loss_siamese: torch.float32
            loss_siamese_q: torch.float32
            loss_predictions: tuple(torch.float32, torch.float32, torch.float32)
        """

        state = bn_states[:, self.burn_in_step, ...]
        action = bn_actions[:, self.burn_in_step, ...]
        d_action = action[..., :self.d_action_summed_size]
        c_action = action[..., -self.c_action_size:]

        batch = state.shape[0]

        q_list = [q(state, c_action) for q in self.model_q_list]
        # ([batch, d_action_summed_size], [batch, 1])
        d_q_list = [q[0] for q in q_list]  # [batch, action_size]
        c_q_list = [q[1] for q in q_list]  # [batch, 1]

        d_y, c_y = self._get_y(n_padding_masks=bn_padding_masks[:, self.burn_in_step:, ...],
                               n_obses_list=[bn_obses[:, self.burn_in_step:, ...] for bn_obses in bn_obses_list],
                               n_states=bn_target_states[:, self.burn_in_step:, ...],
                               n_actions=bn_actions[:, self.burn_in_step:, ...],
                               n_rewards=bn_rewards[:, self.burn_in_step:],
                               next_obs_list=next_obs_list,
                               next_state=next_target_state,
                               n_dones=bn_dones[:, self.burn_in_step:],
                               n_mu_probs=bn_mu_probs[:, self.burn_in_step:] if self.use_n_step_is else None)
        #  [batch, 1], [batch, 1]

        loss_q_list = [torch.zeros((batch, 1), device=self.device) for _ in range(self.ensemble_q_num)]
        loss_none_mse = nn.MSELoss(reduction='none')

        if self.d_action_sizes:
            for i in range(self.ensemble_q_num):
                q_single = torch.sum(d_action * d_q_list[i], dim=-1, keepdim=True) / self.d_action_branch_size  # [batch, 1]
                loss_q_list[i] += loss_none_mse(q_single, d_y)

        if self.c_action_size:
            if self.clip_epsilon > 0:
                target_c_q_list = [q(state, c_action)[1] for q in self.model_target_q_list]

                clipped_q_list = [target_c_q_list[i] + torch.clamp(
                    c_q_list[i] - target_c_q_list[i],
                    -self.clip_epsilon,
                    self.clip_epsilon,
                ) for i in range(self.ensemble_q_num)]

                loss_q_a_list = [loss_none_mse(clipped_q, c_y) for clipped_q in clipped_q_list]  # [batch, 1]
                loss_q_b_list = [loss_none_mse(q, c_y) for q in c_q_list]  # [batch, 1]

                for i in range(self.ensemble_q_num):
                    loss_q_list[i] += torch.maximum(loss_q_a_list[i], loss_q_b_list[i])  # [batch, 1]
            else:
                for i in range(self.ensemble_q_num):
                    loss_q_list[i] += loss_none_mse(c_q_list[i], c_y)  # [batch, 1]

        if priority_is is not None:
            loss_q_list = [loss_q * priority_is for loss_q in loss_q_list]

        loss_q_list = [torch.mean(loss) for loss in loss_q_list]

        if self.optimizer_rep:
            self.optimizer_rep.zero_grad()

        for loss_q, opt_q in zip(loss_q_list, self.optimizer_q_list):
            opt_q.zero_grad()
            loss_q.backward(retain_graph=True)

        grads_rep_main = [m.grad.detach() if m.grad is not None else None
                          for m in self.model_rep.parameters()]
        grads_q_main_list = [[m.grad.detach() if m.grad is not None else None for m in q.parameters()]
                             for q in self.model_q_list]

        """ Siamese Representation Learning """
        loss_siamese, loss_siamese_q = None, None
        if self.siamese is not None:
            loss_siamese, loss_siamese_q = self._train_siamese_representation_learning(
                grads_rep_main=grads_rep_main,
                grads_q_main_list=grads_q_main_list,
                bn_indexes=bn_indexes,
                bn_padding_masks=bn_padding_masks,
                bn_obses_list=bn_obses_list,
                bn_actions=bn_actions)

        for opt_q in self.optimizer_q_list:
            opt_q.step()

        """ Recurrent Prediction Model """
        loss_predictions = None
        if self.use_prediction:
            m_obses_list = [torch.cat([n_obses, next_obs.unsqueeze(1)], dim=1)
                            for n_obses, next_obs in zip(bn_obses_list, next_obs_list)]
            m_states = torch.cat([bn_states, next_state.unsqueeze(1)], dim=1)
            m_target_states = torch.cat([bn_target_states, next_target_state.unsqueeze(1)], dim=1)

            loss_predictions = self._train_rpm(grads_rep_main=grads_rep_main,
                                               m_obses_list=m_obses_list,
                                               m_states=m_states,
                                               m_target_states=m_target_states,
                                               bn_actions=bn_actions,
                                               bn_rewards=bn_rewards)

        if self.optimizer_rep:
            self.optimizer_rep.step()

        return loss_q_list[0], loss_siamese, loss_siamese_q, loss_predictions

    @torch.no_grad()
    def calculate_adaptive_weights(self,
                                   grads_main: List[torch.Tensor],
                                   loss_list: List[torch.Tensor],
                                   model: nn.Module) -> None:

        grads_aux_list = [autograd.grad(loss, model.parameters(),
                                        allow_unused=True,
                                        retain_graph=True)
                          for loss in loss_list]
        grads_aux_list = [[g_aux if g_aux is not None else torch.zeros_like(g_main)
                           for g_main, g_aux in zip(grads_main, grads_aux)]
                          for grads_aux in grads_aux_list]

        _grads_main = torch.cat([g.reshape(1, -1) for g in grads_main], dim=1)
        _grads_aux_list = [torch.cat([g.reshape(1, -1) for g in grads_aux], dim=1)
                           for grads_aux in grads_aux_list]

        cos_list = [functional.cosine_similarity(_grads_main, _grads_aux)
                    for _grads_aux in _grads_aux_list]
        cos_list = [torch.sign(cos).clamp(min=0) for cos in cos_list]

        for grads_aux, cos in zip(grads_aux_list, cos_list):
            for param, grad_aux in zip(model.parameters(), grads_aux):
                param.grad += cos * grad_aux

    def _train_siamese_representation_learning(self,
                                               grads_rep_main: List[torch.Tensor],
                                               grads_q_main_list: List[List[torch.Tensor]],
                                               bn_indexes: torch.Tensor,
                                               bn_padding_masks: torch.Tensor,
                                               bn_obses_list: List[torch.Tensor],
                                               bn_actions: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                  Optional[torch.Tensor]]:
        """
        Args:
            grads_rep_main list(torch.Tensor)
            grads_q_main_list list(list(torch.Tensor))
            bn_indexes (torch.int32): [batch, b + n]
            bn_padding_masks (torch.bool): [batch, b + n]
            bn_obses_list: list([batch, b + n, *obs_shapes_i], ...)
            bn_actions: [batch, b + n, action_size]

        Returns:
            loss_siamese
            loss_siamese_q
        """

        n_padding_masks = bn_padding_masks[:, self.burn_in_step:]
        n_obses_list = [bn_obses[:, self.burn_in_step:, ...] for bn_obses in bn_obses_list]
        encoder_list = self.model_rep.get_augmented_encoders(n_obses_list)  # [batch, n, f], ...
        target_encoder_list = self.model_target_rep.get_augmented_encoders(n_obses_list)  # [batch, n, f], ...

        if not isinstance(encoder_list, tuple):
            encoder_list = (encoder_list, )
            target_encoder_list = (target_encoder_list, )

        batch, n, *_ = encoder_list[0].shape

        if self.siamese == SIAMESE.ATC:
            _encoder_list = [e.reshape(batch * n, -1) for e in encoder_list]  # [batch * n, f], ...
            _target_encoder_list = [t_e.reshape(batch * n, -1) for t_e in target_encoder_list]  # [batch * n, f], ...
            logits_list = [torch.mm(e, weight) for e, weight in zip(_encoder_list, self.contrastive_weight_list)]
            logits_list = [torch.mm(logits, t_e.t()) for logits, t_e in zip(logits_list, _target_encoder_list)]  # [batch * n, batch * n], ...
            contrastive_labels = torch.block_diag(*torch.ones(batch, n, n, device=self.device))  # [batch * n, batch * n]

            padding_mask = n_padding_masks.reshape(batch * n, 1)  # [batch * n, 1]

            loss_siamese_list = [functional.binary_cross_entropy_with_logits(logits,
                                                                             contrastive_labels,
                                                                             reduction='none')
                                 for logits in logits_list]
            loss_siamese_list = [(loss * padding_mask).mean() for loss in loss_siamese_list]

        elif self.siamese == SIAMESE.BYOL:
            _encoder_list = [e.reshape(batch * n, -1) for e in encoder_list]  # [batch * n, f], ...
            projection_list = [pro(encoder) for pro, encoder in zip(self.model_rep_projection_list, _encoder_list)]
            prediction_list = [pre(projection) for pre, projection in zip(self.model_rep_prediction_list, projection_list)]
            _target_encoder_list = [t_e.reshape(batch * n, -1) for t_e in target_encoder_list]  # [batch * n, f], ...
            t_projection_list = [t_pro(t_e) for t_pro, t_e in zip(self.model_target_rep_projection_list, _target_encoder_list)]

            padding_mask = n_padding_masks.reshape(batch * n)  # [batch * n, ]

            loss_siamese_list = [(functional.cosine_similarity(prediction, t_projection) * padding_mask).mean()  # [batch * n, ] -> [1, ]
                                 for prediction, t_projection in zip(prediction_list, t_projection_list)]

        if self.siamese_use_q:
            if self.seq_encoder is None:
                _obs = [n_obses[:, 0, ...] for n_obses in n_obses_list]

                _encoder = [e[:, 0, ...] for e in encoder_list]
                _target_encoder = [t_e[:, 0, ...] for t_e in target_encoder_list]

                state = self.model_rep.get_state_from_encoders(_obs,
                                                               _encoder if len(_encoder) > 1 else _encoder[0])
                target_state = self.model_target_rep.get_state_from_encoders(_obs,
                                                                             _target_encoder if len(_target_encoder) > 1 else _target_encoder[0])

            else:
                obses_list_at_n = [n_obses[:, 0:1, ...] for n_obses in n_obses_list]

                _encoder = [e[:, 0:1, ...] for e in encoder_list]
                _target_encoder = [t_e[:, 0:1, ...] for t_e in target_encoder_list]

                pre_actions_at_n = bn_actions[:, self.burn_in_step - 1:self.burn_in_step, ...]

                if self.seq_encoder == SEQ_ENCODER.RNN:
                    padding_masks_at_n = bn_padding_masks[:, self.burn_in_step:self.burn_in_step + 1]
                    state = self.model_rep.get_state_from_encoders(obses_list_at_n,
                                                                   _encoder if len(_encoder) > 1 else _encoder[0],
                                                                   pre_actions_at_n,
                                                                   self.get_initial_seq_hidden_state(batch, False),
                                                                   padding_mask=padding_masks_at_n)
                    target_state = self.model_target_rep.get_state_from_encoders(obses_list_at_n,
                                                                                 _target_encoder if len(_target_encoder) > 1 else _target_encoder[0],
                                                                                 pre_actions_at_n,
                                                                                 self.get_initial_seq_hidden_state(batch, False),
                                                                                 padding_mask=padding_masks_at_n)
                    state = state[:, 0, ...]
                    target_state = target_state[:, 0, ...]

                elif self.seq_encoder == SEQ_ENCODER.ATTN:
                    indexes_at_n = bn_indexes[:, self.burn_in_step:self.burn_in_step + 1]
                    padding_masks_at_n = bn_padding_masks[:, self.burn_in_step:self.burn_in_step + 1]
                    state = self.model_rep.get_state_from_encoders(indexes_at_n,
                                                                   obses_list_at_n,
                                                                   _encoder if len(_encoder) > 1 else _encoder[0],
                                                                   pre_actions_at_n,
                                                                   query_length=1,
                                                                   padding_mask=padding_masks_at_n)
                    target_state = self.model_target_rep.get_state_from_encoders(indexes_at_n,
                                                                                 obses_list_at_n,
                                                                                 _encoder if len(_encoder) > 1 else _encoder[0],
                                                                                 pre_actions_at_n,
                                                                                 query_length=1,
                                                                                 padding_mask=padding_masks_at_n)
                    state = state[:, 0, ...]
                    target_state = target_state[:, 0, ...]

            q_loss_list = []

            d_action = bn_actions[:, self.burn_in_step, :self.d_action_summed_size]
            c_action = bn_actions[:, self.burn_in_step, -self.c_action_size:]

            q_list = [q(state, c_action)
                      for q in self.model_q_list]  # ([batch, d_action_summed_size], [batch, 1]), ...
            target_q_list = [q(target_state, c_action)
                             for q in self.model_target_q_list]  # ([batch, d_action_summed_size], [batch, 1]), ...

            if self.d_action_sizes:
                q_single_list = [torch.sum(d_action * q[0], dim=-1) / self.d_action_branch_size
                                 for q in q_list]
                # [batch, d_action_summed_size], ... -> [batch, ], ...
                target_q_single_list = [torch.sum(d_action * t_q[0], dim=-1) / self.d_action_branch_size
                                        for t_q in target_q_list]
                # [batch, d_action_summed_size], ... -> [batch, ], ...

                q_loss_list += [functional.mse_loss(q, t_q)
                                for q, t_q in zip(q_single_list, target_q_single_list)]

            if self.c_action_size:
                c_q_list = [q[1] for q in q_list]  # [batch, 1], ...
                target_c_q_list = [t_q[1] for t_q in target_q_list]  # [batch, 1], ...

                q_loss_list += [functional.mse_loss(q, t_q)
                                for q, t_q in zip(c_q_list, target_c_q_list)]

            loss_list = loss_siamese_list + q_loss_list
        else:
            loss_list = loss_siamese_list

        if self.siamese_use_q:
            if self.siamese_use_adaptive:
                for grads_q_main, q_loss, q in zip(grads_q_main_list, q_loss_list, self.model_q_list):
                    self.calculate_adaptive_weights(grads_q_main, [q_loss], q)
            else:
                for q_loss, q in zip(q_loss_list, self.model_q_list):
                    q_loss.backward(inputs=list(q.parameters()), retain_graph=True)

        loss = sum(loss_list)

        if self.siamese_use_adaptive:
            self.calculate_adaptive_weights(grads_rep_main, loss_list, self.model_rep)
        else:
            loss.backward(inputs=list(self.model_rep.parameters()), retain_graph=True)

        self.optimizer_siamese.zero_grad()
        if self.siamese == SIAMESE.ATC:
            loss.backward(inputs=self.contrastive_weight_list, retain_graph=True)
        elif self.siamese == SIAMESE.BYOL:
            loss.backward(inputs=list(chain(*[pro.parameters() for pro in self.model_rep_projection_list],
                                            *[pre.parameters() for pre in self.model_rep_prediction_list])), retain_graph=True)
        self.optimizer_siamese.step()

        return sum(loss_siamese_list), sum(q_loss_list) if self.siamese_use_q else None

    def _train_rpm(self,
                   grads_rep_main,
                   m_obses_list,
                   m_states,
                   m_target_states,
                   bn_actions,
                   bn_rewards) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bn_obses_list = [m_obs[:, :-1, ...] for m_obs in m_obses_list]
        bn_states = m_states[:, :-1, ...]

        approx_next_state_dist: torch.distributions.Normal = self.model_transition(
            [bn_obses[:, self.burn_in_step:, ...] for bn_obses in bn_obses_list],  # May for extra observations
            bn_states[:, self.burn_in_step:, ...],
            bn_actions[:, self.burn_in_step:, ...]
        )  # [batch, n, action_size]

        loss_transition = -torch.mean(approx_next_state_dist.log_prob(m_target_states[:, self.burn_in_step + 1:, ...]))

        std_normal = distributions.Normal(torch.zeros_like(approx_next_state_dist.loc),
                                          torch.ones_like(approx_next_state_dist.scale))
        kl = distributions.kl.kl_divergence(approx_next_state_dist, std_normal)
        loss_transition += self.transition_kl * torch.mean(kl)

        approx_n_rewards = self.model_reward(m_states[:, self.burn_in_step + 1:, ...])  # [batch, n, 1]
        loss_reward = functional.mse_loss(approx_n_rewards, torch.unsqueeze(bn_rewards[:, self.burn_in_step:], 2))
        loss_reward /= self.n_step

        loss_obs = self.model_observation.get_loss(m_states[:, self.burn_in_step:, ...],
                                                   [m_obses[:, self.burn_in_step:, ...] for m_obses in m_obses_list])
        loss_obs /= self.n_step

        self.calculate_adaptive_weights(grads_rep_main, [loss_transition, loss_reward, loss_obs], self.model_rep)

        loss_predictions = loss_transition + loss_reward + loss_obs
        self.optimizer_prediction.zero_grad()
        loss_predictions.backward(inputs=list(chain(self.model_transition.parameters(),
                                                    self.model_reward.parameters(),
                                                    self.model_observation.parameters())))
        self.optimizer_prediction.step()

        return torch.mean(approx_next_state_dist.entropy()), loss_reward, loss_obs

    def _train_policy(self,
                      obs_list: List[torch.Tensor],
                      state: torch.Tensor,
                      action: torch.Tensor,
                      mu_d_policy_probs: torch.Tensor = None) -> Tuple[Optional[torch.Tensor],
                                                                       Optional[torch.Tensor]]:
        batch = state.shape[0]

        d_policy, c_policy = self.model_policy(state, obs_list)

        loss_d_policy = torch.zeros((batch, 1), device=self.device)
        loss_c_policy = torch.zeros((batch, 1), device=self.device)

        with torch.no_grad():
            d_alpha = torch.exp(self.log_d_alpha)
            c_alpha = torch.exp(self.log_c_alpha)

        if self.d_action_sizes and not self.discrete_dqn_like:
            probs = d_policy.probs   # [batch, d_action_summed_size]
            clipped_probs = probs.clamp(min=1e-8)

            c_action = action[..., -self.c_action_size:]

            q_list = [q(state, c_action) for q in self.model_q_list]
            # ([batch, d_action_summed_size], [batch, 1])
            d_q_list = [q[0] for q in q_list]  # [batch, d_action_summed_size]

            stacked_d_q = torch.stack(d_q_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, batch, d_action_summed_size] -> [ensemble_q_sample, batch, d_action_summed_size]
            mean_d_q = torch.mean(stacked_d_q, dim=0)
            # [ensemble_q_sample, batch, d_action_summed_size] -> [batch, d_action_summed_size]

            _loss_policy = d_alpha * torch.log(clipped_probs) - mean_d_q.detach()  # [batch, d_action_summed_size]
            loss_d_policy = torch.sum(probs * _loss_policy, dim=1, keepdim=True) / self.d_action_branch_size  # [batch, 1]

            clipped_mu_d_policy_probs = mu_d_policy_probs.clamp(min=1e-8)
            mu_d_policy_entropy = -torch.sum(mu_d_policy_probs * torch.log(clipped_mu_d_policy_probs), dim=-1) / self.d_action_branch_size  # [batch, ]
            pi_d_policy_entropy = d_policy.entropy().sum(-1) / self.d_action_branch_size  # [batch, ]
            entropy_penalty = torch.pow(mu_d_policy_entropy - pi_d_policy_entropy, 2.) / 2.   # [batch, ]
            loss_d_policy += self.d_policy_entropy_penalty * entropy_penalty.unsqueeze(-1)  # [batch, 1]

        if self.c_action_size:
            action_sampled = c_policy.rsample()
            c_q_for_gradient_list = [q(state, torch.tanh(action_sampled))[1] for q in self.model_q_list]
            # [[batch, 1], ...]

            stacked_c_q_for_gradient = torch.stack(c_q_for_gradient_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, batch, 1] -> [ensemble_q_sample, batch, 1]

            log_prob = torch.sum(squash_correction_log_prob(c_policy, action_sampled), dim=1, keepdim=True)
            # [batch, 1]

            min_c_q_for_gradient, _ = torch.min(stacked_c_q_for_gradient, dim=0)
            # [ensemble_q_sample, batch, 1] -> [batch, 1]

            loss_c_policy = c_alpha * log_prob - min_c_q_for_gradient
            # [batch, 1]

        loss_policy = torch.mean(loss_d_policy + loss_c_policy)

        if self.use_pr:
            # obs_list[2]:[256,2][是否绿灯，是否等待区域]
            # action: [256,2][steer方向，motor速度]
            regularization_term = self.llm_wrapper.get_value(obs_list[2], action)
            loss_policy = loss_policy - regularization_term

        if (self.d_action_sizes and not self.discrete_dqn_like) or self.c_action_size:
            self.optimizer_policy.zero_grad()
            loss_policy.backward(inputs=list(self.model_policy.parameters()))
            self.optimizer_policy.step()

        return (torch.mean(d_policy.entropy().sum(-1) / self.d_action_branch_size) if self.d_action_sizes else None,
                torch.mean(c_policy.entropy()) if self.c_action_size else None)

    def _train_alpha(self,
                     obs_list: torch.Tensor,
                     state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = state.shape[0]

        with torch.no_grad():
            d_policy, c_policy = self.model_policy(state, obs_list)

        loss_d_alpha = torch.zeros((batch, 1), device=self.device)
        loss_c_alpha = torch.zeros((batch, 1), device=self.device)

        if self.d_action_sizes and not self.discrete_dqn_like:
            probs = d_policy.probs   # [batch, d_action_summed_size]
            clipped_probs = probs.clamp(min=1e-8)

            _loss_alpha = -self.log_d_alpha * (torch.log(clipped_probs) + self.target_d_alpha)  # [batch, d_action_summed_size]
            loss_d_alpha = torch.sum(probs * _loss_alpha, dim=1, keepdim=True) / self.d_action_branch_size  # [batch, 1]

        if self.c_action_size:
            action_sampled = c_policy.sample()
            log_prob = torch.sum(squash_correction_log_prob(c_policy, action_sampled), dim=1, keepdim=True)
            # [batch, 1]

            loss_c_alpha = -self.log_c_alpha * (log_prob + self.target_c_alpha)  # [batch, 1]

        loss_alpha = torch.mean(loss_d_alpha + loss_c_alpha)

        self.optimizer_alpha.zero_grad()
        loss_alpha.backward(inputs=[self.log_d_alpha, self.log_c_alpha])
        self.optimizer_alpha.step()

        d_alpha = torch.exp(self.log_d_alpha)
        c_alpha = torch.exp(self.log_c_alpha)

        return d_alpha, c_alpha

    def _train_curiosity(self,
                         bn_padding_masks: torch.Tensor,
                         m_states: torch.Tensor,
                         bn_actions: torch.Tensor) -> torch.Tensor:
        n_padding_masks = bn_padding_masks[:, self.burn_in_step:]
        n_states = m_states[:, self.burn_in_step:-1]
        next_n_states = m_states[:, self.burn_in_step + 1:]
        n_actions = bn_actions[:, self.burn_in_step:]

        self.optimizer_curiosity.zero_grad()

        if self.curiosity == CURIOSITY.FORWARD:
            approx_next_n_states = self.model_forward_dynamic(n_states, n_actions)
            loss_curiosity = functional.mse_loss(approx_next_n_states, next_n_states, reduction='none')
            loss_curiosity *= ~bn_padding_masks.unsqueeze(-1)
            loss_curiosity = torch.mean(loss_curiosity)
            loss_curiosity.backward(inputs=list(self.model_forward_dynamic.parameters()))

        elif self.curiosity == CURIOSITY.INVERSE:
            approx_n_actions = self.model_inverse_dynamic(n_states, next_n_states)
            loss_curiosity = functional.mse_loss(approx_n_actions, n_actions, reduction='none')
            loss_curiosity *= ~bn_padding_masks.unsqueeze(-1)
            loss_curiosity = torch.mean(loss_curiosity)
            loss_curiosity.backward(inputs=list(self.model_inverse_dynamic.parameters()))

        self.optimizer_curiosity.step()

        return loss_curiosity

    def _train_rnd(self,
                   bn_padding_masks: torch.Tensor,
                   bn_states: torch.Tensor,
                   bn_actions: torch.Tensor) -> torch.Tensor:
        n_padding_masks = bn_padding_masks[:, self.burn_in_step:]
        n_states = bn_states[:, self.burn_in_step:]
        n_actions = bn_actions[:, self.burn_in_step:]
        d_n_actions = n_actions[..., :self.d_action_summed_size]  # [batch, n, d_action_summed_size]
        c_n_actions = n_actions[..., -self.c_action_size:]  # [batch, n, c_action_size]

        loss = torch.zeros((1, ), device=self.device)

        if self.d_action_sizes:
            d_rnd = self.model_rnd.cal_d_rnd(n_states)  # [batch, n, d_action_summed_size, f]
            with torch.no_grad():
                t_d_rnd = self.model_target_rnd.cal_d_rnd(n_states)  # [batch, n, d_action_summed_size, f]

            _i = d_n_actions.unsqueeze(-1)  # [batch, n, d_action_summed_size, 1]
            d_rnd = (torch.repeat_interleave(_i, d_rnd.shape[-1], dim=-1) * d_rnd).sum(-2)
            # [batch, n, d_action_summed_size, f] -> [batch, n, f]
            t_d_rnd = (torch.repeat_interleave(_i, t_d_rnd.shape[-1], dim=-1) * t_d_rnd).sum(-2)
            # [batch, n, d_action_summed_size, f] -> [batch, n, f]

            _loss = functional.mse_loss(d_rnd, t_d_rnd, reduction='none')
            _loss *= ~n_padding_masks.unsqueeze(-1)
            loss += torch.mean(_loss)

        if self.c_action_size:
            c_rnd = self.model_rnd.cal_c_rnd(n_states, c_n_actions)  # [batch, n, f]
            with torch.no_grad():
                t_c_rnd = self.model_target_rnd.cal_c_rnd(n_states, c_n_actions)  # [batch, n, f]

            _loss = functional.mse_loss(c_rnd, t_c_rnd, reduction='none')
            _loss *= ~n_padding_masks.unsqueeze(-1)
            loss += torch.mean(_loss)

        self.optimizer_rnd.zero_grad()
        loss.backward(inputs=list(self.model_rnd.parameters()))
        self.optimizer_rnd.step()

        return loss

    def _train(self,
               bn_indexes: torch.Tensor,
               bn_padding_masks: torch.Tensor,
               bn_obses_list: List[torch.Tensor],
               bn_actions: torch.Tensor,
               bn_rewards: torch.Tensor,
               next_obs_list: List[torch.Tensor],
               bn_dones: torch.Tensor,
               bn_mu_probs: torch.Tensor,
               f_seq_hidden_states: torch.Tensor = None,
               priority_is: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            bn_indexes (torch.int32): [batch, b + n]
            bn_padding_masks (torch.bool): [batch, b + n]
            bn_obses_list: list([batch, b + n, *obs_shapes_i], ...)
            bn_actions: [batch, b + n, action_size]
            bn_rewards: [batch, b + n]
            next_obs_list: list([batch, *obs_shapes_i], ...)
            bn_dones (torch.bool): [batch, b + n]
            bn_mu_probs: [batch, b + n, action_size]
            f_seq_hidden_states: [batch, 1, *seq_hidden_state_shape]
            priority_is: [batch, 1]

        Returns:
            m_target_states: [batch, b + n + 1, state_size]
        """

        if self.global_step % self.update_target_per_step == 0:
            self._update_target_variables(tau=self.tau)

        (m_indexes,
         m_padding_masks,
         m_obses_list,
         m_pre_actions) = self.get_m_data(bn_indexes=bn_indexes,
                                          bn_padding_masks=bn_padding_masks,
                                          bn_obses_list=bn_obses_list,
                                          bn_actions=bn_actions,
                                          next_obs_list=next_obs_list)

        m_states, _ = self.get_l_states(l_indexes=m_indexes,
                                        l_padding_masks=m_padding_masks,
                                        l_obses_list=m_obses_list,
                                        l_pre_actions=m_pre_actions,
                                        f_seq_hidden_states=f_seq_hidden_states,
                                        is_target=False)

        m_target_states, _ = self.get_l_states(l_indexes=m_indexes,
                                               l_padding_masks=m_padding_masks,
                                               l_obses_list=m_obses_list,
                                               l_pre_actions=m_pre_actions,
                                               f_seq_hidden_states=f_seq_hidden_states,
                                               is_target=True)

        (loss_q,
         loss_siamese,
         loss_siamese_q,
         loss_predictions) = self._train_rep_q(bn_indexes=bn_indexes,
                                               bn_padding_masks=bn_padding_masks,
                                               bn_obses_list=bn_obses_list,
                                               bn_states=m_states[:, :-1, ...],
                                               bn_target_states=m_target_states[:, :-1, ...],
                                               bn_actions=bn_actions,
                                               bn_rewards=bn_rewards,
                                               next_obs_list=next_obs_list,
                                               next_state=m_states[:, -1, ...],
                                               next_target_state=m_target_states[:, -1, ...],
                                               bn_dones=bn_dones,
                                               bn_mu_probs=bn_mu_probs,
                                               priority_is=priority_is)

        with torch.no_grad():
            m_states, _ = self.get_l_states(l_indexes=m_indexes,
                                            l_padding_masks=m_padding_masks,
                                            l_obses_list=m_obses_list,
                                            l_pre_actions=m_pre_actions,
                                            f_seq_hidden_states=f_seq_hidden_states,
                                            is_target=False)

        obs_list = [m_obses[:, self.burn_in_step, ...] for m_obses in m_obses_list]
        state = m_states[:, self.burn_in_step, ...]
        action = bn_actions[:, self.burn_in_step, ...]
        mu_d_policy_probs = bn_mu_probs[:, self.burn_in_step, :self.d_action_summed_size]

        d_policy_entropy, c_policy_entropy = self._train_policy(obs_list, state, action,
                                                                mu_d_policy_probs)

        if self.use_auto_alpha and ((self.d_action_sizes and not self.discrete_dqn_like) or self.c_action_size):
            d_alpha, c_alpha = self._train_alpha(obs_list, state)

        if self.curiosity is not None:
            loss_curiosity = self._train_curiosity(bn_padding_masks, m_states, bn_actions)

        if self.use_rnd:
            bn_states = m_states[:, :-1, ...]
            loss_rnd = self._train_rnd(bn_padding_masks, bn_states, bn_actions)

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            self.summary_available = True

            with torch.no_grad():
                self.summary_writer.add_scalar('loss/q', loss_q, self.global_step)
                if self.d_action_sizes and not self.discrete_dqn_like:
                    self.summary_writer.add_scalar('loss/d_entropy', d_policy_entropy, self.global_step)
                    if self.use_auto_alpha and not self.discrete_dqn_like:
                        self.summary_writer.add_scalar('loss/d_alpha', d_alpha, self.global_step)
                if self.c_action_size:
                    self.summary_writer.add_scalar('loss/c_entropy', c_policy_entropy, self.global_step)
                    if self.use_auto_alpha:
                        self.summary_writer.add_scalar('loss/c_alpha', c_alpha, self.global_step)

                if self.siamese is not None:
                    self.summary_writer.add_scalar('loss/siamese',
                                                   loss_siamese,
                                                   self.global_step)
                    if self.siamese_use_q:
                        self.summary_writer.add_scalar('loss/siamese_q',
                                                       loss_siamese_q,
                                                       self.global_step)

                if self.use_prediction:
                    approx_next_state_dist_entropy, loss_reward, loss_obs = loss_predictions
                    self.summary_writer.add_scalar('loss/transition',
                                                   approx_next_state_dist_entropy,
                                                   self.global_step)
                    self.summary_writer.add_scalar('loss/reward', loss_reward, self.global_step)
                    self.summary_writer.add_scalar('loss/observation', loss_obs, self.global_step)

                    approx_obs_list = self.model_observation(m_states[0:1, 0, ...])
                    if not isinstance(approx_obs_list, (list, tuple)):
                        approx_obs_list = [approx_obs_list]
                    for approx_obs in approx_obs_list:
                        if len(approx_obs.shape) > 3:
                            self.summary_writer.add_images('observation',
                                                           approx_obs.permute([0, 3, 1, 2]),
                                                           self.global_step)

                if self.curiosity is not None:
                    self.summary_writer.add_scalar('loss/curiosity', loss_curiosity, self.global_step)

                if self.use_rnd:
                    self.summary_writer.add_scalar('loss/rnd', loss_rnd, self.global_step)

            self.summary_writer.flush()

        return m_target_states

    @torch.no_grad()
    def _get_td_error(self,
                      bn_padding_masks: torch.Tensor,
                      bn_obses_list: List[torch.Tensor],
                      bn_states: torch.Tensor,
                      bn_target_states: torch.Tensor,
                      bn_actions: torch.Tensor,
                      bn_rewards: torch.Tensor,
                      next_obs_list: List[torch.Tensor],
                      next_target_state: torch.Tensor,
                      bn_dones: torch.Tensor,
                      bn_mu_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bn_padding_masks (torch.bool): [batch, b + n]
            bn_obses_list: list([batch, b + n, *obs_shapes_i], ...)
            bn_states: [batch, b + n, state_size]
            bn_target_states: [batch, b + n, state_size]
            bn_actions: [batch, b + n, action_size]
            bn_rewards: [batch, b + n]
            next_obs_list: list([batch, *obs_shapes_i], ...)
            next_target_state: [batch, state_size]
            bn_dones (torch.bool): [batch, b + n]
            bn_mu_probs: [batch, b + n, action_size]

        Returns:
            The td-error of observations, [batch, 1]
        """
        state = bn_states[:, self.burn_in_step, ...]
        action = bn_actions[:, self.burn_in_step, ...]
        d_action = action[..., :self.d_action_summed_size]
        c_action = action[..., -self.c_action_size:]

        q_list = [q(state, c_action) for q in self.model_q_list]
        # ([batch, d_action_summed_size], [batch, 1])
        d_q_list = [q[0] for q in q_list]  # [batch, d_action_summed_size]
        c_q_list = [q[1] for q in q_list]  # [batch, 1]

        if self.d_action_sizes:
            d_q_list = [torch.sum(d_action * q, dim=-1, keepdim=True) / self.d_action_branch_size
                        for q in d_q_list]
            # [batch, 1]

        d_y, c_y = self._get_y(n_padding_masks=bn_padding_masks[:, self.burn_in_step:, ...],
                               n_obses_list=[bn_obses[:, self.burn_in_step:, ...] for bn_obses in bn_obses_list],
                               n_states=bn_target_states[:, self.burn_in_step:, ...],
                               n_actions=bn_actions[:, self.burn_in_step:, ...],
                               n_rewards=bn_rewards[:, self.burn_in_step:],
                               next_obs_list=next_obs_list,
                               next_state=next_target_state,
                               n_dones=bn_dones[:, self.burn_in_step:],
                               n_mu_probs=bn_mu_probs[:, self.burn_in_step:] if self.use_n_step_is else None)
        # [batch, 1], [batch, 1]

        q_td_error_list = [torch.zeros((state.shape[0], 1), device=self.device) for _ in range(self.ensemble_q_num)]
        # [batch, 1]
        if self.d_action_sizes:
            for i in range(self.ensemble_q_num):
                q_td_error_list[i] += torch.abs(d_q_list[i] - d_y)

        if self.c_action_size:
            for i in range(self.ensemble_q_num):
                q_td_error_list[i] += torch.abs(c_q_list[i] - c_y)

        td_error = torch.mean(torch.cat(q_td_error_list, dim=-1),
                              dim=-1, keepdim=True)
        return td_error

    def get_episode_td_error(self,
                             l_indexes: np.ndarray,
                             l_padding_masks: np.ndarray,
                             l_obses_list: List[np.ndarray],
                             l_actions: np.ndarray,
                             l_rewards: np.ndarray,
                             l_dones: np.ndarray,
                             l_probs: np.ndarray = None,
                             l_seq_hidden_states: np.ndarray = None) -> torch.Tensor:
        """
        Args:
            l_indexes: [1, ep_len]
            l_padding_masks (bool): [1, ep_len]
            l_obses_list: list([1, ep_len, *obs_shapes_i], ...)
            l_actions: [1, ep_len, action_size]
            l_rewards: [1, ep_len]
            l_dones: [1, ep_len]
            l_probs: [1, ep_len, action_size]
            l_seq_hidden_states: [1, ep_len, *seq_hidden_state_shape]

        Returns:
            The td-error of raw episode observations
            [ep_len, ]
        """
        (bn_indexes,
         bn_padding_masks,
         bn_obses_list,
         bn_actions,
         bn_rewards,
         next_obs_list,
         bn_dones,
         bn_probs,
         f_seq_hidden_states) = episode_to_batch(burn_in_step=self.burn_in_step,
                                                 n_step=self.n_step,
                                                 padding_action=self._padding_action,
                                                 l_indexes=l_indexes,
                                                 l_padding_masks=l_padding_masks,
                                                 l_obses_list=l_obses_list,
                                                 l_actions=l_actions,
                                                 l_rewards=l_rewards,
                                                 l_dones=l_dones,
                                                 l_probs=l_probs,
                                                 l_seq_hidden_states=l_seq_hidden_states)

        """
        bn_indexes: [ep_len - bn + 1, bn]
        bn_padding_masks: [ep_len - bn + 1, bn]
        bn_obses_list: list([ep_len - bn + 1, bn, *obs_shapes_i], ...)
        bn_actions: [ep_len - bn + 1, bn, action_size]
        bn_rewards: [ep_len - bn + 1, bn]
        next_obs_list: list([ep_len - bn + 1, *obs_shapes_i], ...)
        bn_dones: [ep_len - bn + 1, bn]
        bn_probs: [ep_len - bn + 1, bn, action_size]
        f_seq_hidden_states: [ep_len - bn + 1, 1, *seq_hidden_state_shape]
        """

        td_error_list = []
        all_batch = bn_obses_list[0].shape[0]
        batch_size = self.batch_size
        for i in range(math.ceil(all_batch / batch_size)):
            b_i, b_j = i * batch_size, (i + 1) * batch_size

            _bn_indexes = torch.from_numpy(bn_indexes[b_i:b_j]).to(self.device)
            _bn_padding_masks = torch.from_numpy(bn_padding_masks[b_i:b_j]).to(self.device)
            _bn_obses_list = [torch.from_numpy(o[b_i:b_j]).to(self.device) for o in bn_obses_list]
            _bn_actions = torch.from_numpy(bn_actions[b_i:b_j]).to(self.device)
            _bn_rewards = torch.from_numpy(bn_rewards[b_i:b_j]).to(self.device)
            _next_obs_list = [torch.from_numpy(o[b_i:b_j]).to(self.device) for o in next_obs_list]
            _bn_dones = torch.from_numpy(bn_dones[b_i:b_j]).to(self.device)
            _bn_probs = torch.from_numpy(bn_probs[b_i:b_j]).to(self.device)
            _f_seq_hidden_states = torch.from_numpy(f_seq_hidden_states[b_i:b_j]).to(self.device) if self.seq_encoder is not None else None

            (_m_indexes,
             _m_padding_masks,
             _m_obses_list,
             _m_pre_actions) = self.get_m_data(bn_indexes=_bn_indexes,
                                               bn_padding_masks=_bn_padding_masks,
                                               bn_obses_list=_bn_obses_list,
                                               bn_actions=_bn_actions,
                                               next_obs_list=_next_obs_list)

            _m_states, _ = self.get_l_states(l_indexes=_m_indexes,
                                             l_padding_masks=_m_padding_masks,
                                             l_obses_list=_m_obses_list,
                                             l_pre_actions=_m_pre_actions,
                                             f_seq_hidden_states=_f_seq_hidden_states if self.seq_encoder is not None else None,
                                             is_target=False)

            _m_target_states, _ = self.get_l_states(l_indexes=_m_indexes,
                                                    l_padding_masks=_m_padding_masks,
                                                    l_obses_list=_m_obses_list,
                                                    l_pre_actions=_m_pre_actions,
                                                    f_seq_hidden_states=_f_seq_hidden_states if self.seq_encoder is not None else None,
                                                    is_target=True)

            td_error = self._get_td_error(bn_padding_masks=_bn_padding_masks,
                                          bn_obses_list=_bn_obses_list,
                                          bn_states=_m_states[:, :-1, ...],
                                          bn_target_states=_m_target_states[:, :-1, ...],
                                          bn_actions=_bn_actions,
                                          bn_rewards=_bn_rewards,
                                          next_obs_list=_next_obs_list,
                                          next_target_state=_m_target_states[:, -1, ...],
                                          bn_dones=_bn_dones,
                                          bn_mu_probs=_bn_probs).detach().cpu().numpy()
            td_error_list.append(td_error.flatten())

        td_error = np.concatenate([*td_error_list,
                                   np.zeros(1, dtype=np.float32)])
        return td_error

    def log_episode(self, **episode_trans: np.ndarray) -> None:
        if self.summary_writer is None or not self.summary_available:
            return

        ep_indexes = episode_trans['ep_indexes']
        ep_obses_list = episode_trans['ep_obses_list']
        ep_actions = episode_trans['ep_actions']

        if self.seq_encoder == SEQ_ENCODER.ATTN:
            with torch.no_grad():
                pre_l_actions = gen_pre_n_actions(ep_actions)
                *_, attn_weights_list = self.model_rep(torch.from_numpy(ep_indexes).to(self.device),
                                                       [torch.from_numpy(o).to(self.device) for o in ep_obses_list],
                                                       pre_action=torch.from_numpy(pre_l_actions).to(self.device),
                                                       query_length=ep_indexes.shape[1])

                for i, attn_weight in enumerate(attn_weights_list):
                    image = plot_attn_weight(attn_weight[0].cpu().numpy())
                    self.summary_writer.add_figure(f'attn_weight/{i}', image, self.global_step)

        self.summary_available = False

    def put_episode(self, **episode_trans: np.ndarray) -> None:
        # Ignore episodes which length is too short
        if episode_trans['ep_indexes'].shape[1] < self.n_step:
            return

        episodes = self._padding_next_obs_list(**episode_trans)

        if self.use_replay_buffer:
            self._fill_replay_buffer(*episodes)
        else:
            self.batch_buffer.put_episode(*episodes)

    def _padding_next_obs_list(self,
                               ep_indexes: np.ndarray,
                               ep_obses_list: List[np.ndarray],
                               ep_actions: np.ndarray,
                               ep_rewards: np.ndarray,
                               next_obs_list: List[np.ndarray],
                               ep_dones: np.ndarray,
                               ep_probs: List[np.ndarray],
                               ep_seq_hidden_states: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        """
        Padding next_obs_list

        Args:
            ep_indexes (np.int32): [1, ep_len]
            ep_obses_list (np): list([1, ep_len, *obs_shapes_i], ...)
            ep_actions (np): [1, ep_len, action_size]
            ep_rewards (np): [1, ep_len]
            next_obs_list (np): list([1, *obs_shapes_i], ...)
            ep_dones (bool): [1, ep_len]
            ep_probs (np): [1, ep_len, action_size]
            ep_seq_hidden_states (np): [1, ep_len, *seq_hidden_state_shape]

        Returns:
            ep_indexes (np.int32): [1, ep_len + 1]
            ep_padding_masks: (bool): [1, ep_len + 1]
            ep_obses_list (np): list([1, ep_len + 1, *obs_shapes_i], ...)
            ep_actions (np): [1, ep_len + 1, action_size]
            ep_rewards (np): [1, ep_len + 1]
            ep_dones (bool): [1, ep_len + 1]
            ep_probs (np): [1, ep_len + 1, action_size]
            ep_seq_hidden_states (np): [1, ep_len + 1, *seq_hidden_state_shape]
        """

        ep_indexes = np.concatenate([ep_indexes,
                                     ep_indexes[:, -1:] + 1], axis=1)
        ep_padding_masks = np.zeros_like(ep_indexes, dtype=bool)
        ep_padding_masks[:, -1] = True
        ep_obses_list = [np.concatenate([obs, np.expand_dims(next_obs, 1)], axis=1)
                         for obs, next_obs in zip(ep_obses_list, next_obs_list)]
        ep_actions = np.concatenate([ep_actions,
                                     self._padding_action.reshape(1, 1, -1)], axis=1)
        ep_rewards = np.concatenate([ep_rewards,
                                     np.zeros([1, 1], dtype=np.float32)], axis=1)
        ep_dones = np.concatenate([ep_dones,
                                   np.ones([1, 1], dtype=bool)], axis=1)
        ep_probs = np.concatenate([ep_probs,
                                  np.ones([1, 1, ep_probs.shape[-1]], dtype=np.float32)], axis=1)

        if ep_seq_hidden_states is not None:
            ep_seq_hidden_states = np.concatenate([ep_seq_hidden_states,
                                                   np.zeros([1, 1, *ep_seq_hidden_states.shape[2:]], dtype=np.float32)], axis=1)

        return (
            ep_indexes,
            ep_padding_masks,
            ep_obses_list,
            ep_actions,
            ep_rewards,
            ep_dones,
            ep_probs,
            ep_seq_hidden_states if ep_seq_hidden_states is not None else None
        )

    def _fill_replay_buffer(self,
                            ep_indexes: np.ndarray,
                            ep_padding_masks: np.ndarray,
                            ep_obses_list: List[np.ndarray],
                            ep_actions: np.ndarray,
                            ep_rewards: np.ndarray,
                            ep_dones: np.ndarray,
                            ep_probs: List[np.ndarray],
                            ep_seq_hidden_states: Optional[np.ndarray] = None) -> None:
        """
        Args:
            ep_indexes (np.int32): [1, ep_len]
            ep_padding_masks: (bool): [1, ep_len]
            ep_obses_list (np): list([1, ep_len, *obs_shapes_i], ...)
            ep_actions (np): [1, ep_len, action_size]
            ep_rewards (np): [1, ep_len]
            ep_dones (bool): [1, ep_len]
            ep_probs (np): [1, ep_len, action_size]
            ep_seq_hidden_states (np): [1, ep_len, *seq_hidden_state_shape]
        """

        # Reshape [1, ep_len, ...] to [ep_len, ...]
        index = ep_indexes.squeeze(0)
        padding_mask = ep_padding_masks.squeeze(0)
        obs_list = [ep_obses.squeeze(0) for ep_obses in ep_obses_list]
        if self.use_normalization:
            self._udpate_normalizer([torch.from_numpy(obs).to(self.device) for obs in obs_list])
        action = ep_actions.squeeze(0)
        reward = ep_rewards.squeeze(0)
        done = ep_dones.squeeze(0)
        mu_prob = ep_probs.squeeze(0)

        storage_data = {
            'index': index,
            'padding_mask': padding_mask,
            **{f'obs_{name}': obs for name, obs in zip(self.obs_names, obs_list)},
            'action': action,
            'reward': reward,
            'done': done,
            'mu_prob': mu_prob
        }

        if ep_seq_hidden_states is not None:
            seq_hidden_state = ep_seq_hidden_states.squeeze(0)
            storage_data['seq_hidden_state'] = seq_hidden_state

        # n_step transitions except the first one and the last obs
        self.replay_buffer.add(storage_data, ignore_size=1)

    def _sample_from_replay_buffer(self) -> Tuple[np.ndarray,
                                                  Tuple[Union[np.ndarray, List[np.ndarray]], ...]]:
        """
        Sample from replay buffer

        Returns:
            pointers: [batch, ]
            (
                bn_indexes (np.int32): [batch, b + n]
                bn_padding_masks (bool): [batch, b + n]
                bn_obses_list (np): list([batch, b + n, *obs_shapes_i], ...)
                bn_actions (np): [batch, b + n, action_size]
                bn_rewards (np): [batch, b + n]
                next_obs_list (np): list([batch, *obs_shapes_i], ...)
                bn_dones (np): [batch, b + n]
                bn_mu_probs (np): [batch, b + n, action_size]
                bn_seq_hidden_states (np): [batch, b + n, *seq_hidden_state_shape],
                priority_is (np): [batch, 1]
            )
        """
        sampled = self.replay_buffer.sample()
        if sampled is None:
            return None

        """
        trans:
            index (np.int32): [batch, ]
            padding_mask (bool): [batch, ]
            obs_i: [batch, *obs_shapes_i]
            action: [batch, action_size]
            reward: [batch, ]
            done (bool): [batch, ]
            mu_prob: [batch, action_size]
            seq_hidden_state: [batch, *seq_hidden_state_shape]
        """
        pointers, trans, priority_is = sampled

        # Get burn_in_step + n_step transitions
        # TODO: could be faster, no need get all data
        batch = {k: np.zeros((v.shape[0],
                              self.burn_in_step + self.n_step + 1,
                              *v.shape[1:]), dtype=v.dtype)
                 for k, v in trans.items()}

        for k, v in trans.items():
            batch[k][:, self.burn_in_step] = v

        def set_padding(t, mask):
            t['index'][mask] = -1
            t['padding_mask'][mask] = True
            for n in self.obs_names:
                t[f'obs_{n}'][mask] = 0.
            t['action'][mask] = self._padding_action
            t['reward'][mask] = 0.
            t['done'][mask] = True
            t['mu_prob'][mask] = 1.
            if 'seq_hidden_state' in t:
                t['seq_hidden_state'][mask] = 0.

        # Get next n_step data
        for i in range(1, self.n_step + 1):
            t_trans = self.replay_buffer.get_storage_data(pointers + i)

            mask = (t_trans['index'] - trans['index']) != i
            set_padding(t_trans, mask)

            for k, v in t_trans.items():
                batch[k][:, self.burn_in_step + i] = v

        # Get previous burn_in_step data
        for i in range(self.burn_in_step):
            t_trans = self.replay_buffer.get_storage_data(pointers - i - 1)

            mask = (trans['index'] - t_trans['index']) != i + 1
            set_padding(t_trans, mask)

            for k, v in t_trans.items():
                batch[k][:, self.burn_in_step - i - 1] = v

        """
        m_indexes (np.int32): [batch, N + 1]
        m_padding_masks (bool): [batch, N + 1]
        m_obses_list: list([batch, N + 1, *obs_shapes_i], ...)
        m_actions: [batch, N + 1, action_size]
        m_rewards: [batch, N + 1]
        m_dones (bool): [batch, N + 1]
        m_mu_probs: [batch, N + 1, action_size]
        m_seq_hidden_state: [batch, N + 1, *seq_hidden_state_shape]
        """
        m_indexes = batch['index']
        m_padding_masks = batch['padding_mask']
        m_obses_list = [batch[f'obs_{name}'] for name in self.obs_names]
        m_actions = batch['action']
        m_rewards = batch['reward']
        m_dones = batch['done']
        m_mu_probs = batch['mu_prob']

        bn_indexes = m_indexes[:, :-1]
        bn_padding_masks = m_padding_masks[:, :-1]
        bn_obses_list = [m_obses[:, :-1, ...] for m_obses in m_obses_list]
        bn_actions = m_actions[:, :-1, ...]
        bn_rewards = m_rewards[:, :-1]
        next_obs_list = [m_obses[:, -1, ...] for m_obses in m_obses_list]
        bn_dones = m_dones[:, :-1]
        bn_mu_probs = m_mu_probs[:, :-1]

        if self.seq_encoder is not None:
            m_seq_hidden_states = batch['seq_hidden_state']
            bn_seq_hidden_states = m_seq_hidden_states[:, :-1, ...]

        return pointers, (bn_indexes,
                          bn_padding_masks,
                          bn_obses_list,
                          bn_actions,
                          bn_rewards,
                          next_obs_list,
                          bn_dones,
                          bn_mu_probs,
                          bn_seq_hidden_states if self.seq_encoder is not None else None,
                          priority_is if self.use_priority else None)

    @unified_elapsed_timer('train_all', 10)
    def train(self, train_all_profiler) -> int:
        step = self.get_global_step()

        # 检查是否使用重放缓冲区replay_buffer，以决定从哪个数据源中提取训练样本
        if self.use_replay_buffer:
            with self._profiler('sample_from_replay_buffer', repeat=10) as profiler:
                train_data = self._sample_from_replay_buffer()
                if train_data is None:
                    profiler.ignore()
                    train_all_profiler.ignore()
                    return step

            pointers, batch = train_data
            batch_list = [batch]
        else:
            batch_list = self.batch_buffer.get_batch()
            batch_list = [(*batch, None) for batch in batch_list]  # None is priority_is

        for batch in batch_list:
            (bn_indexes,
             bn_padding_masks,
             bn_obses_list,
             bn_actions,
             bn_rewards,
             next_obs_list,
             bn_dones,
             bn_mu_probs,
             bn_seq_hidden_states,
             priority_is) = batch

            """
            bn_indexes (np.int32): [batch, b + n] (256, 33)
            bn_padding_masks (bool): [batch, b + n]
            bn_obses_list: list([batch, b + n, *obs_shapes_i], ...)
            bn_actions: [batch, b + n, action_size]
            bn_rewards: [batch, b + n] (256, 33)
            next_obs_list: list([batch, *obs_shapes_i], ...)
            bn_dones (bool): [batch, b + n]
            bn_mu_probs: [batch, b + n, action_size] (256, 33, 2)
            bn_seq_hidden_states: [batch, b + n, *seq_hidden_state_shape]
            priority_is: [batch, 1] (256, 1)
            """
            # 将训练批次中的各种数据从 NumPy 数组转换为 PyTorch 张量，并将其移动到指定的计算设备 GPU
            with self._profiler('to gpu', repeat=10):
                bn_indexes = torch.from_numpy(bn_indexes).to(self.device)
                bn_padding_masks = torch.from_numpy(bn_padding_masks).to(self.device)
                bn_obses_list = [torch.from_numpy(t).to(self.device) for t in bn_obses_list]
                for i, bn_obses in enumerate(bn_obses_list):
                    # obs is image. It is much faster to convert uint8 to float32 in GPU
                    if bn_obses.dtype == torch.uint8:
                        bn_obses_list[i] = bn_obses.type(torch.float32) / 255.
                bn_actions = torch.from_numpy(bn_actions).to(self.device)
                bn_rewards = torch.from_numpy(bn_rewards).to(self.device)
                next_obs_list = [torch.from_numpy(t).to(self.device) for t in next_obs_list]
                for i, next_obs in enumerate(next_obs_list):
                    # obs is image
                    if next_obs.dtype == torch.uint8:
                        next_obs_list[i] = next_obs.type(torch.float32) / 255.
                bn_dones = torch.from_numpy(bn_dones).to(self.device)
                bn_mu_probs = torch.from_numpy(bn_mu_probs).to(self.device)
                if self.seq_encoder is not None:
                    f_seq_hidden_states = bn_seq_hidden_states[:, :1]
                    f_seq_hidden_states = torch.from_numpy(f_seq_hidden_states).to(self.device)
                if self.use_replay_buffer and self.use_priority:
                    priority_is = torch.from_numpy(priority_is).to(self.device)

            with self._profiler('train', repeat=10):
                m_target_states = self._train(
                    bn_indexes=bn_indexes,
                    bn_padding_masks=bn_padding_masks,
                    bn_obses_list=bn_obses_list,
                    bn_actions=bn_actions,
                    bn_rewards=bn_rewards,
                    next_obs_list=next_obs_list,
                    bn_dones=bn_dones,
                    bn_mu_probs=bn_mu_probs,
                    f_seq_hidden_states=f_seq_hidden_states if self.seq_encoder is not None else None,
                    priority_is=priority_is if self.use_replay_buffer and self.use_priority else None)

            if step % self.save_model_per_step == 0:
                self.save_model()

            if self.use_replay_buffer:
                bn_pre_actions = gen_pre_n_actions(bn_actions)  # [batch, b + n, action_size]

                with self._profiler('get_l_states_with_seq_hidden_states', repeat=10):
                    bn_states, next_bn_seq_hidden_states = self.get_l_states_with_seq_hidden_states(
                        l_indexes=bn_indexes,
                        l_padding_masks=bn_padding_masks,
                        l_obses_list=bn_obses_list,
                        l_pre_actions=bn_pre_actions,
                        f_seq_hidden_states=f_seq_hidden_states if self.seq_encoder is not None else None)

                if self.use_n_step_is or (self.d_action_sizes and not self.discrete_dqn_like):
                    with self._profiler('get_l_probs', repeat=10):
                        bn_pi_probs_tensor = self.get_l_probs(
                            l_obses_list=bn_obses_list,
                            l_states=bn_states,
                            l_actions=bn_actions)

                # Update td_error
                if self.use_priority:
                    with self._profiler('get_td_error', repeat=10):
                        td_error = self._get_td_error(
                            bn_padding_masks=bn_padding_masks,
                            bn_obses_list=bn_obses_list,
                            bn_states=bn_states,
                            bn_target_states=m_target_states[:, :-1, ...],
                            bn_actions=bn_actions,
                            bn_rewards=bn_rewards,
                            next_obs_list=next_obs_list,
                            next_target_state=m_target_states[:, -1, ...],
                            bn_dones=bn_dones,
                            bn_mu_probs=bn_pi_probs_tensor).detach().cpu().numpy()
                    self.replay_buffer.update(pointers, td_error)

                bn_padding_masks = bn_padding_masks.detach().cpu().numpy()
                padding_mask = bn_padding_masks.reshape(-1)

                # Update seq_hidden_states
                if self.seq_encoder is not None:
                    pointers_list = [pointers + 1 + i for i in range(-self.burn_in_step, self.n_step)]
                    tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)

                    next_bn_seq_hidden_states = next_bn_seq_hidden_states.detach().cpu().numpy()
                    seq_hidden_state = next_bn_seq_hidden_states.reshape(-1, *next_bn_seq_hidden_states.shape[2:])
                    self.replay_buffer.update_transitions(tmp_pointers[~padding_mask], 'seq_hidden_state', seq_hidden_state[~padding_mask])

                # Update n_mu_probs
                if self.use_n_step_is or (self.d_action_sizes and not self.discrete_dqn_like):
                    pointers_list = [pointers + i for i in range(-self.burn_in_step, self.n_step)]
                    tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)

                    pi_probs = bn_pi_probs_tensor.detach().cpu().numpy()
                    pi_prob = pi_probs.reshape(-1, *pi_probs.shape[2:])
                    self.replay_buffer.update_transitions(tmp_pointers[~padding_mask], 'mu_prob', pi_prob[~padding_mask])

            step = self._increase_global_step()

        return step
