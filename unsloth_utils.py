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


# Offloading to disk for modules (lm_head, embed_tokens)
import pickle

BUFFERS_SAVED_DIR = "_unsloth_temporary_saved_buffers"

def offload_to_disk(W, model, name, temporary_location : str = BUFFERS_SAVED_DIR):
    file_location = os.path.join(temporary_location, model.config._name_or_path)
    if not os.path.exists(file_location):
        os.makedirs(file_location)
    pass

    filename = os.path.join(file_location, f"{name}.pt")
    W = W.weight if hasattr(W, "weight") else W
    torch.save(W, filename, pickle_module = pickle, pickle_protocol = pickle.HIGHEST_PROTOCOL,)
    offloaded_W = torch.load(filename, map_location = "cpu", mmap = True)
    offloaded_W._offloaded_file_location = filename
    return offloaded_W
pass


def offload_input_embeddings(model, temporary_location : str = BUFFERS_SAVED_DIR):
    offloaded_W = offload_to_disk(
        model.get_input_embeddings(), 
        model, 
        "input_embeddings", 
        temporary_location,
    )
    new_input_embeddings = torch.nn.Embedding.from_pretrained(offloaded_W)
    new_input_embeddings._offloaded_file_location = offloaded_W._offloaded_file_location
    model.set_input_embeddings(new_input_embeddings)
    return
pass


def offload_output_embeddings(model, temporary_location : str = BUFFERS_SAVED_DIR):
    offloaded_W = offload_to_disk(
        model.get_output_embeddings(), 
        model, 
        "output_embeddings", 
        temporary_location,
    )

    new_output_embeddings = torch.nn.Linear(1, 1, bias = None)
    del new_output_embeddings.weight

    new_output_embeddings.weight = offloaded_W
    new_output_embeddings.in_features  = offloaded_W.shape[1]
    new_output_embeddings.out_features = offloaded_W.shape[0]

    new_output_embeddings._offloaded_file_location = offloaded_W._offloaded_file_location
    model.set_output_embeddings(new_output_embeddings)
    return
pass


def unsloth_patch_embeddings_and_lm_head(model):
    # First offload lm_head and embed_tokens to disk
    input_embeddings_device  = model. get_input_embeddings().weight.device
    output_embeddings_device = model.get_output_embeddings().weight.device

    print("Unsloth: Offloading input_embeddings to disk to save VRAM")
    offload_input_embeddings(model)

    print("Unsloth: Offloading output_embeddings to disk to save VRAM")
    offload_output_embeddings(model)

    # Remove old items to save VRAM
    for _ in range(3): gc.collect(); torch.cuda.empty_cache()

    # Now patch lm_head and embed_tokens
    print("Unsloth: Casting embed_tokens to float32")
    # print(model.model)
    assert(hasattr(model.model.embed_tokens, "modules_to_save"))
    model.model.embed_tokens.modules_to_save.default.to(
        device = input_embeddings_device, 
        dtype = torch.float32, 
        non_blocking = True,
    )
    model.model.embed_tokens.modules_to_save.default.requires_grad_(True)

    print("Unsloth: Casting lm_head to float32")
    assert(hasattr(model.model.lm_head, "modules_to_save"))
    model.lm_head.modules_to_save.default.to(
        device = output_embeddings_device, 
        dtype = torch.float32, 
        non_blocking = True
    )
    model.lm_head.modules_to_save.default.requires_grad_(True)
