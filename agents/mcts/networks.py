from asyncore import compact_traceback
import jax
import flax.linen as nn

uniform_initializer = nn.initializers.variance_scaling(distribution='uniform', mode='fan_out', scale=0.333)

class PolicyValue(nn.Module):
    layer_sizes: tuple
    num_action: int

    @nn.compact
    def __call__(self, x):
        for feature in self.layer_sizes:
            x = nn.Dense(features=feature, kernel_init=uniform_initializer)(x)
            x = nn.elu(x)

        value = nn.Dense(features=1, kernel_init=uniform_initializer)(x)
        logits = nn.Dense(features=self.num_action, kernel_init=uniform_initializer)(x)

        return logits, value
