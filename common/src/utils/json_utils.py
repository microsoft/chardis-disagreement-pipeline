import jsonstrip
import re


def repair_json(raw: str) -> str:
    """
    Attempt to repair malformed JSON returned by an LLM.

    Repairs performed:
    1. Replace Python-style None with null
    2. Extract the JSON array region if wrapped in extra text
    3. Strip invalid JSON comments
    4. Remove illegal escape characters
    """

    # 1. Convert Python None → JSON null
    json_str = raw.replace("None", "null")

    # 2. Extract the bracketed array if extra text surrounds it
    open_brace_index = json_str.find("[")
    close_brace_index = json_str.rfind("]")

    if open_brace_index != -1 and close_brace_index != -1:
        json_str = json_str[open_brace_index: close_brace_index + 1]

    # 3. Strip invalid comments (using jsonstrip)
    try:
        json_str = jsonstrip.strip(json_str)
    except Exception:
        # Best effort — continue if jsonstrip fails
        pass

    # 4. Remove invalid escape sequences
    json_str = re.sub(
        r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})',
        "",
        json_str
    )

    return json_str
