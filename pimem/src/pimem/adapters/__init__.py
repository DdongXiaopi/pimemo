from .pi import PiAdapter

__all__ = ["GenericMemoryAdapter", "PiAdapter"]


def __getattr__(name):
    if name == "GenericMemoryAdapter":
        from .generic import GenericMemoryAdapter
        return GenericMemoryAdapter
    raise AttributeError(name)
