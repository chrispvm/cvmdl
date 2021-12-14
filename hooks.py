import torch

from . import utils

# To use the Hooks class, inherit from it and define any subset of the following:
# Simply create an instance of that class, and it will automatically add the hooks to its module.


hooks = [
    # These apply the hook to each leaf module in self.module
    "backward_hook",  # Function signature: (self, module_of_hook, input_grad, output_grad)
    "forward_hook",  # Function signature: (self, module_of_hook, input, output)
    "forward_pre_hook",  # Function signature: (self, module_of_hook, input)
    # These apply the hook to self.module
    "model_backward_hook",  # Function signature: (self, module_of_hook, input_grad, output_grad)
    "model_forward_hook",  # Function signature: (self, module_of_hook, input, output)
    "model_forward_pre_hook",  # Function signature: (self, module_of_hook, input)
    # This is to the input tensor of each leaf module in self.module:
    "hook_on_output_tensor"]


class Hooks:
    def __init__(self, module):
        self.module = module
        self._named_modules = utils.named_flatten_module(module)
        self._removable_handles = []
        self._register_hooks()

    def _register_hooks(self):
        for name in ["backward_hook", "forward_hook", "forward_pre_hook"]:
            self._register_model_hook_by_name(name)
            self._register_hook_by_name(name)

        if hasattr(self, "hook_on_output_tensor"):
            # print("has hook_on_output_tensor. len(modules):", len(self._named_modules))
            for name, m in self._named_modules:
                self._register_hook(m, "forward_hook", self._hook_register_hook_on_output_tensor)

    def full_hook_name(self, hook_name):
        if hook_name == "backward_hook": return "full_backward_hook"
        return hook_name

    def _register_hook_by_name(self, hook_name):
        hook = getattr(self, hook_name, False)
        if hook:
            for name, m in self._named_modules:
                self._register_hook(m, self.full_hook_name(hook_name), hook)
                # print("full_hook_name: ", self.full_hook_name(hook_name))

    def _register_model_hook_by_name(self, hook_name):
        hook = getattr(self, f"model_{hook_name}", False)
        if hook: self._register_hook(self.module, self.full_hook_name(hook_name), hook)

    def _register_hook(self, m, hook_name, hook):
        register_fn = getattr(m, f"register_{hook_name}")
        rh = register_fn(hook)
        self._removable_handles.append(rh)

    def _remove_hooks(self):
        # print("removing hooks")
        if self._removable_handles:
            for rh in self._removable_handles: rh.remove()

    def flush_caches(self):
        self.remove()

    def remove(self):
        self._remove_hooks()
        self._named_modules = None
        self._removable_handles = None
        self.module = None

    def __del__(self):  # upon garbage collection, it will remove the hooks.
        self.remove()

    def activate(self):
        assert not self._removable_handles
        self._register_hooks()

    def deactivate(self):
        self._remove_hooks()

    @torch.utils.hooks.unserializable_hook
    def _hook_register_hook_on_output_tensor(self, m, inp, outp):
        if not self.module.training: return

        @torch.utils.hooks.unserializable_hook
        def _fn(grad):
            _hook = getattr(self, f"hook_on_output_tensor")
            return _hook(m, grad)

        rh = outp.register_hook(_fn)
        self._removable_handles.append(rh)


class PlotGradientDistributionHooks(Hooks):
    def backward_hook(self, m, inp, outp):
        print(m)


class PrintEverythingHooks(Hooks):
    def backward_hook(self, m, in_grad, out_grad):
        print("backward (model): ", m)
        print("backward (inp_grad): ", in_grad)
        print("backward (out_grad): ", out_grad)
        print(" ")

    def forward_hook(self, m, inp, outp):
        print("forward (model): ", m)
        print("forward (inp): ", inp)
        print("forward (out): ", outp)
        print(" ")

    def forward_pre_hook(self, m, inp):
        print("forward_pre (model): ", m)
        print("forward_pre (inp): ", inp)
        print(" ")

    def model_backward_hook(self, m, in_grad, out_grad):
        print("model_backward (model): ", m)
        print("model_backward (inp_grad): ", in_grad)
        print("model_backward (out_grad): ", out_grad)
        print(" ")

    def model_forward_hook(self, m, inp, outp):
        print("model_forward (model): ", m)
        print("model_forward (inp): ", inp)
        print("model_forward (out): ", outp)
        print(" ")

    def model_forward_pre_hook(self, m, inp):
        print("model_forward_pre (model): ", m)
        print("model_forward_pre (inp): ", inp)
        print(" ")
