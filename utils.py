
import ast
import logging

log = logging.getLogger(__name__)

NO_TEMPORAL_INFO_MSG = 'No temporal information found'

def safe_literal_eval(x):
    """
    Safely evaluate a string as a Python literal.
    Returns the evaluated object if possible; if evaluation fails or input is empty, returns the original input unchanged.
    Logs a warning if evaluation fails.
    Args:
        x (str): Input to evaluate.
    Returns:
        Any: Evaluated Python object, or original input if evaluation fails or input is empty.
    """
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return x  # Return empty string as is
        elif NO_TEMPORAL_INFO_MSG in x:
            return x
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError) as e:
            log.warning(f"Could not parse string: {x}. Exception: {e}")
            return x
    return x