"""Exception types. Exceptions are permitted only at configuration boundaries
(AD-10); the residual path returns saturated values / status objects instead."""


class ConfigError(ValueError):
    """Invalid user-supplied configuration or geometry input."""