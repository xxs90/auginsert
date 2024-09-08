"""
Config for BC algorithm.
"""

from robomimic.config.base_config import BaseConfig


class BCConfig(BaseConfig):
    ALGO_NAME = "bc"

    def train_config(self):
        """
        BC algorithms don't need "next_obs" from hdf5 - so save on storage and compute by disabling it.
        """
        super(BCConfig, self).train_config()
        self.train.hdf5_load_next_obs = False

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.policy.learning_rate.scheduler_type = "multistep" # learning rate scheduler ("multistep", "linear", etc) 
        self.algo.optim_params.policy.regularization.L2 = 0.00          # L2 regularization strength

        # loss weights
        self.algo.loss.l2_weight = 1.0      # L2 loss weight
        self.algo.loss.l1_weight = 0.0      # L1 loss weight
        self.algo.loss.cos_weight = 0.0     # cosine loss weight

        # MLP network architecture (layers after observation encoder and RNN, if present)
        self.algo.actor_layer_dims = (1024, 1024)

        # RNN policy settings
        self.algo.rnn.enabled = False                               # whether to train RNN policy
        self.algo.rnn.horizon = 10                                  # unroll length for RNN - should usually match train.seq_length
        self.algo.rnn.hidden_dim = 400                              # hidden dimension size    
        self.algo.rnn.rnn_type = "LSTM"                             # rnn type - one of "LSTM" or "GRU"
        self.algo.rnn.num_layers = 2                                # number of RNN layers that are stacked
        self.algo.rnn.open_loop = False                             # if True, action predictions are only based on a single observation (not sequence)
        self.algo.rnn.kwargs.bidirectional = False                  # rnn kwargs
        self.algo.rnn.kwargs.do_not_lock_keys()

        # Transformer policy settings
        self.algo.transformer.enabled = False                       # whether to train transformer policy
        self.algo.transformer.context_length = 10                   # length of (s, a) seqeunces to feed to transformer - should usually match train.frame_stack
        self.algo.transformer.embed_dim = 512                       # dimension for embeddings used by transformer
        self.algo.transformer.num_layers = 6                        # number of transformer blocks to stack
        self.algo.transformer.num_heads = 8                         # number of attention heads for each transformer block (should divide embed_dim evenly)
        self.algo.transformer.emb_dropout = 0.1                     # dropout probability for embedding inputs in transformer
        self.algo.transformer.attn_dropout = 0.1                    # dropout probability for attention outputs for each transformer block
        self.algo.transformer.block_output_dropout = 0.1            # dropout probability for final outputs for each transformer block
        self.algo.transformer.sinusoidal_embedding = False          # if True, use standard positional encodings (sin/cos)
        self.algo.transformer.activation = "gelu"                   # activation function for MLP in Transformer Block
        self.algo.transformer.supervise_all_steps = False           # if true, supervise all intermediate actions, otherwise only final one
        self.algo.transformer.nn_parameter_for_timesteps = True     # if true, use nn.Parameter otherwise use nn.Embedding

        # VTT encoder settings
        self.algo.vtt.enabled = False                               # whether to encode inputs with VTT
        self.algo.vtt.modality_independent_vtt = True               # specify the type of VTT architecture 
        self.algo.vtt.clip_gradients = False                        # clip grad norms to 1.0 during training

        self.algo.vtt.vtt_kwargs.img_sizes = (84,)
        self.algo.vtt.vtt_kwargs.img_patch_size = 14
        self.algo.vtt.vtt_kwargs.tactile_dim = 12
        self.algo.vtt.vtt_kwargs.tactile_patches = 2
        self.algo.vtt.vtt_kwargs.tactile_history = 32
        self.algo.vtt.vtt_kwargs.in_channels = 3
        self.algo.vtt.vtt_kwargs.embed_dim = 384
        self.algo.vtt.vtt_kwargs.output_dim = 288
        self.algo.vtt.vtt_kwargs.depth = 6
        self.algo.vtt.vtt_kwargs.num_heads = 8
        self.algo.vtt.vtt_kwargs.mlp_ratio = 4.0
        self.algo.vtt.vtt_kwargs.qkv_bias = False
        self.algo.vtt.vtt_kwargs.qk_scale = False
        self.algo.vtt.vtt_kwargs.drop_rate = 0.0
        self.algo.vtt.vtt_kwargs.attn_drop_rate = 0.0
        self.algo.vtt.vtt_kwargs.drop_path_rate = 0.0

        # Modality-independent VTT specific parameters
        self.algo.vtt.vtt_kwargs.num_latents = 64
        self.algo.vtt.vtt_kwargs.depth_vtt = 3
        self.algo.vtt.vtt_kwargs.depth_latent = 3
        self.algo.vtt.vtt_kwargs.token_drop_rate = 0.4
