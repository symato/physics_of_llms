# source https://github.com/unslothai/unsloth/blob/main/unsloth/models/_utils.py
import torch
import os
import gc

from packaging.version import Version
if Version(torch.__version__) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = "cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = "cuda")
pass

class Unsloth_Offloaded_Gradient_Checkpointer(torch.autograd.Function):
    """
    Saves VRAM by smartly offloading to RAM.
    Tiny hit to performance, since we mask the movement via non blocking calls.
    """
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        saved_hidden_states = hidden_states.to("cpu", non_blocking = True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        return output
    pass

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to("cuda", non_blocking = True).detach()
        hidden_states.requires_grad = True
        with torch.enable_grad():
            (output,) = ctx.forward_function(hidden_states, *ctx.args)
        torch.autograd.backward(output, dY)
        return (None, hidden_states.grad,) + (None,)*len(ctx.args)
    pass
pass


@torch._disable_dynamo
def unsloth_offloaded_gradient_checkpoint(function, *args, use_reentrant = None, **kwargs):
    return Unsloth_Offloaded_Gradient_Checkpointer.apply(function, *args)
pass


# source https://github.com/axolotl-ai-cloud/axolotl/tree/main/src/axolotl/utils/gradient_checkpointing
@torch._disable_dynamo
def hf_grad_checkpoint_unsloth_wrapper(
    decoder_layer, *args, use_reentrant=None
):  # pylint: disable=unused-argument
    return Unsloth_Offloaded_Gradient_Checkpointer.apply(
        decoder_layer.__self__,
        *args,
    )
