import importlib


def create_instance(module_name: str, class_name: str, *args, **kwargs):
    """
    Dynamically imports a module and creates an instance of the given class.

    Args:
        module_name: The name of the module to import.
        class_name: The name of the class to instantiate.
        *args: Positional arguments to pass to the class constructor.
        **kwargs: Keyword arguments to pass to the class constructor.

    Returns:
        An instance of the specified class.
    """
    try:
        # Import the module dynamically
        module = importlib.import_module(module_name)

        # Get the class from the module
        cls = getattr(module, class_name)

        # Ensure it's actually a class
        if not isinstance(cls, type):
            raise TypeError(f"{class_name} is not a class in {module_name}")

        # Create and return an instance
        return cls(*args, **kwargs)

    except ModuleNotFoundError:
        raise ImportError(f"Module '{module_name}' not found.")
    except AttributeError:
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_name}'.")
    except Exception as e:
        raise RuntimeError(f"Error creating instance: {e}")
