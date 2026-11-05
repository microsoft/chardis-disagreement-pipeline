from common.src.configs.config_value import ConfigValue

class ConfigurableBase:
    """
    Base class for configurable components.
    """

    def __init__(self):
        pass

    def load_config(self, config: dict, value_name: str, required: bool = True):
        """
        Initializes this instance with values from a configuration dictionary. 
        The values are are properties on objects in the dictionary whose name corresponds to the class attribute.

        Args:
            config: The configuration dictionary.
            value_name: The name of the value in the configuration dictionary that corresponds to this instance.
            required: Whether the configuration values for this instance are required.
        """
        if config is None:
            config = dict()

        instance_config = config.get(value_name, dict())
        errors = list()

        # Get all the attribute values in this instance that are ConfigValue instances
        attributes = vars(self)
        config_values = {name: value for (name, value) in attributes.items() if isinstance(value, ConfigValue)}
        
        # Try to load each config value for each ConfigValue attribute
        for name, config_value in config_values.items():
            setattr(
                self,
                name,
                self._load_config_value(
                    instance_config,
                    name,
                    errors,
                    required=config_value.required and required,
                    default=config_value.default_value,
                    help=config_value.help
                )
            )

        config[value_name] = instance_config

        if len(errors) == 0:
            # No errors loading the config
            return False

        # Since there were errors print each error message
        for error in errors:
            print(error)

        return True

    def _load_config_value(self, config, property_name: str, errors: list, required: bool = False, default=None, help: str = ""):
        """
        Load a configuration value from a config dictionary.

        Args:
            config: The configuration dictionary.
            property: The property name to load.
            required: Whether the property is required.
            default: The default value if the property is not found.
            help: Help text for the property.
        """
        if property_name not in config:
            # Property not specified in the config.
            value = default if default is not None else "<TBD>"

            config[property_name] = {
                "description": help,
                "value": value
            }

            if required:
                # If the property is required then this is an error
                errors.append(
                    f"Missing required config property. {property_name}: {help}")

            return value

        try:
            # Get the specified value
            return config[property_name]["value"]
        except Exception as e:
            # Exception when trying to load the value
            error_message = getattr(e, "message", str(e))
            errors.append(
                f"Error loading config property. {property_name}: {error_message}")

            value = default if default is not None else "<TBD>"
            config[property_name] = {
                "description": help,
                "value": value
            }

            return value
