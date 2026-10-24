"""Exact arithmetic for the model.

Small local models are unreliable at adding up lists of numbers, so the agent
gets a `calculate` tool that does it with Decimal arithmetic instead.
"""
import ast
import re
from decimal import Decimal, DivisionByZero, InvalidOperation, getcontext

getcontext().prec = 40

MAX_ABS = Decimal("1e60")
HINT = (
    "Use plain arithmetic with + - * / ** % and parentheses. "
    "Write percentages as decimals, e.g. 15% of 20000 -> 0.15*20000."
)


def _normalize(text):
    s = str(text).strip()
    s = s.split("=")[0]
    s = re.sub(r"₦|\$|€|£|¥|NGN|USD|EUR|GBP", "", s, flags=re.I)
    s = re.sub(r"\bN(?=\d)", "", s)
    s = s.replace("×", "*").replace("÷", "/").replace("^", "**").replace("−", "-")
    s = re.sub(r"(?<=\d)\s*[x]\s*(?=\d)", "*", s)
    # Thousand separators (1,500 -> 1500), but not argument lists like max(10,100).
    if not re.search(r"[A-Za-z_]\w*\s*\(", s):
        s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*k\b", r"(\1*1000)", s, flags=re.I)
    return s.strip()


def _to_decimal(value):
    d = Decimal(str(value))
    if not d.is_finite():
        raise ValueError("Number is too large.")
    return d


def _flatten(values):
    for v in values:
        if isinstance(v, list):
            yield from _flatten(v)
        else:
            yield v


def _sum(*a):
    return sum(_flatten(a), Decimal(0))


def _avg(*a):
    items = list(_flatten(a))
    if not items:
        raise ValueError("avg() needs at least one number.")
    return sum(items, Decimal(0)) / len(items)


def _round(x, n=0):
    n = int(n)
    return round(x, n)


FUNCTIONS = {
    "sum": _sum,
    "avg": _avg,
    "mean": _avg,
    "min": lambda *a: min(_flatten(a)),
    "max": lambda *a: max(_flatten(a)),
    "abs": lambda x: abs(x),
    "round": _round,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numbers are allowed.")
        return _to_decimal(node.value)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left), _eval(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            result = left + right
        elif isinstance(op, ast.Sub):
            result = left - right
        elif isinstance(op, ast.Mult):
            result = left * right
        elif isinstance(op, ast.Div):
            result = left / right
        elif isinstance(op, ast.FloorDiv):
            result = left // right
        elif isinstance(op, ast.Mod):
            result = left % right
        elif isinstance(op, ast.Pow):
            if abs(right) > 100:
                raise ValueError("Exponent is too large.")
            result = left ** right
        else:
            raise ValueError("Unsupported operator.")
        if abs(result) > MAX_ABS:
            raise ValueError("Result is too large.")
        return result
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.lower() in FUNCTIONS
        and not node.keywords
    ):
        return FUNCTIONS[node.func.id.lower()](*[_eval(a) for a in node.args])
    raise ValueError("Unsupported expression.")


def _plain(d):
    d = d.quantize(Decimal("0.0000000001")) if d != d.to_integral_value() else d
    text = format(d, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _pretty(d):
    plain = _plain(d)
    places = len(plain.split(".")[1]) if "." in plain else 0
    return format(d.quantize(Decimal(1).scaleb(-places)), ",f")


def evaluate(expression):
    """Evaluate one expression. Returns a Decimal; raises ValueError on failure."""
    cleaned = _normalize(expression)
    if not cleaned:
        raise ValueError("Empty expression.")
    try:
        tree = ast.parse(cleaned, mode="eval")
        value = _eval(tree)
    except (SyntaxError, TypeError) as error:
        raise ValueError(f"Could not read '{expression}'. {HINT}") from error
    except (DivisionByZero, InvalidOperation) as error:
        raise ValueError("Division by zero or invalid operation.") from error
    if isinstance(value, list):
        value = _sum(value)
    return value


def calculate(expression=None, expressions=None):
    """Tool entry point.

    expressions: list of expressions, one per line item -> each value plus the grand total.
    expression:  a single expression.
    """
    if expressions is not None:
        if isinstance(expressions, str):
            expressions = [e for e in re.split(r"[\n;]+", expressions) if e.strip()]
        if not isinstance(expressions, list) or not expressions:
            return {"error": "expressions must be a non-empty list of strings."}
        items, total = [], Decimal(0)
        for raw in expressions:
            try:
                value = evaluate(raw)
            except ValueError as error:
                return {"error": f"{error} (while evaluating: {raw!r}). Fix it and call calculate again."}
            total += value
            items.append({"expression": str(raw), "value": _plain(value), "formatted": _pretty(value)})
        return {
            "count": len(items),
            "results": items,
            "total": _plain(total),
            "total_formatted": _pretty(total),
        }

    if expression is None:
        return {"error": "Provide 'expressions' (a list) or 'expression' (a string)."}
    try:
        value = evaluate(expression)
    except ValueError as error:
        return {"error": str(error)}
    return {
        "expression": str(expression),
        "result": _plain(value),
        "result_formatted": _pretty(value),
    }
