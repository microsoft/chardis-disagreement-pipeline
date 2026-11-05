class ConfigValue:
    """
    Represents a ConfigValue that can be used to dynamically generate and load configuration values for a class attribute.

    Args:
        default_value: The default value for the configuration.
        required: Whether this configuration value is required.
        help: Help text describing the configuration value.
    """
    def __init__(self, default_value, required: bool, help: str):
        self.default_value = default_value
        self.required = required
        self.help = help