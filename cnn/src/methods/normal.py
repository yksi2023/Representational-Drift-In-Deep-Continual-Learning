from src.methods.base import BaseContinualMethod


class NormalMethod(BaseContinualMethod):
    """Standard fine-tuning without any continual learning mechanism.

    Uses the base training loop unchanged (plain task loss).
    """
    pass
