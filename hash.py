
# import pyhash
# my_hash = fp = pyhash.farm_fingerprint_128()

# def hash_architecture_of(module):
#     return hash_architecture(module.__class__)


# def hash_architecture(module_class):
#     assert issubclass(module_class, nn.Module)
#     s_init = inspect.getsource(module_class.__init__)
#     s_forward = inspect.getsource(module_class._forward)
#     s = f"{s_init} {s_forward}"
#     return my_hash(s)


# def hash_method(method):
#     method_src = inspect.getsource(method)
#     return my_hash(method_src)


# def hash_method_with_parameters(method, parameters=None):
#     if parameters is None:
#         return hash_method(method)
#     else:
#         method_src = inspect.getsource(method)
#         parameters_src = repr(parameters)
#         return my_hash(f"{method_src}{parameters_src}")