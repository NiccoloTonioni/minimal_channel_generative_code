"""
Parsing of gin config files with the file, line and offending text reported on errors.
"""
import tokenize
import gin


def _locate_error(e):
    """Extract (line, column) from a gin/tokenize parsing error, if available."""
    if isinstance(e, tokenize.TokenError) and len(e.args) > 1:
        return e.args[1]
    if isinstance(e, SyntaxError):  # gin's ParseError subclasses SyntaxError
        return e.lineno, e.offset
    return None, None


def parse_config_file(gin_file):
    """Parse a gin file, reporting the file, line and offending text on failure."""
    try:
        gin.parse_config_file(gin_file)
    except Exception as e:
        lineno, col = _locate_error(e)
        msg = f"Error while parsing gin file '{gin_file}'"
        if lineno is not None:
            msg += f" at line {lineno}, column {col}"
            with open(gin_file) as f:
                lines = f.readlines()
            if 0 < lineno <= len(lines):
                msg += f":\n    {lines[lineno - 1].rstrip()}"
                if col:
                    msg += f"\n    {' ' * max(col - 1, 0)}^"
        msg += f"\n{type(e).__name__}: {e}"
        raise RuntimeError(msg) from e


def parse_config_files(gin_config):
    """Parse a comma-separated list of gin files."""
    for gin_file in gin_config.split(','):
        parse_config_file(gin_file.strip())
