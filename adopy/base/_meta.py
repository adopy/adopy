__all__ = ['MetaInterface']


class MetaInterface(object):
    """
    Meta interface for tasks and models. The class and its inherited classes
    return the same instance when creating new instance.

    .. note::

        This class is for developmental purpose, not intended to be used
        directly by users. If you want to create new task or new model, please
        see :mod:`adopy.base.Task` or :mod:`adopy.base.Model`.

    Parameters
    ----------
    name
        Name value of the class.
    """
    _instance = None  # type: object

    def __init__(self, name: str):
        self._name = name  # type: str

    def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
        # Create new instance if and only if there is no instance created
        # before.
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance

    @property
    def name(self) -> str:
        """Name value of the class."""
        return self._name
