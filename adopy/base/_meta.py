import re
from typing import (
    Any, Callable, Dict, Iterable, Optional, List, Tuple,
    TypeVar
)

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
    name : str
        Name value of the class.
    key : Optional[str]
        Key value for the class. Should be an alphanumeric string that may
        contain ``-`` (hyphen) or ``_`` (underscore).
    """
    _instance = None  # type: object

    def __init__(self, name: str, key: Optional[str] = None):
        self._name = name  # type: str
        if not key:
            new_key = re.sub(r'\s+', '_', name)
            self._key = re.sub(r'[^a-zA-Z0-9_\-]+', '', new_key)
        else:
            self._key = key  # type: str

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

    @property
    def key(self) -> str:
        """
        Key value for the class. If no value is passed, the key will be created
        as an alphanumeric string including ``-`` (hyphen) and ``_``
        (underscore) from the given name. This value is used to check if the
        instance is from the same class.
        """
        return self._key
