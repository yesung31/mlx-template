import importlib
import pkgutil

# Automatically find and import all sub-packages/modules
# and expose their contents here.
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    if module_name == "__init__":
        continue

    # Import the module
    module = importlib.import_module(f".{module_name}", package=__name__)

    # Expose the module's attributes in this package's namespace
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)

        # Simple heuristic: if it's a class and not private, expose it.
        # This allows access like data.TemplateDataModule
        if isinstance(attribute, type) and not attribute_name.startswith("_"):
            globals()[attribute_name] = attribute
