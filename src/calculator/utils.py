import ast
import operator
from datasets import load_dataset, DatasetDict


# Define supported operators
operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def safe_eval(expr: str) -> float | int:
    """
    Safely evaluate a numerical expression with +, -, *, and / using ast.

    Args:
        expr (str): A string containing the mathematical expression to evaluate.

    Returns:
        float or int: The result of the evaluated expression.

    Raises:
        ValueError: If the expression contains unsupported operations or invalid syntax.
    """
    try:
        expr_ast = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression: {expr}") from e

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op_type = type(node.op)

            if op_type in operators:
                op_func = operators[op_type]
                result = op_func(left, right)
                return result
            else:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")

        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                result = +operand
                print(f"Evaluated unary +{operand} = {result}")
                return result
            elif isinstance(node.op, ast.USub):
                result = -operand
                print(f"Evaluated unary -{operand} = {result}")
                return result
            else:
                raise ValueError(
                    f"Unsupported unary operator: {type(node.op).__name__}"
                )

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise ValueError(f"Unsupported constant: {node.value}")

        else:
            raise ValueError(f"Unsupported expression component: {ast.dump(node)}")

    return eval_node(expr_ast.body)


def load_asdiv() -> DatasetDict:
    return load_dataset("yimingzhang/asdiv")


def can_use_calculator(s: str) -> bool:
    """Should we invoke the calculator?

    Args:
        s (str): partial generation

    Returns:
        bool: True iff calculator should be used

    Note: Do not worry about if the expression is valid/well-formed.
          We will check that when using the calculator.

    Hint:
        Q1.2
    """
    return ...


def use_calculator(input: str) -> str:
    """Run calculator on the (potentially not well-formed) string

    If input contains a well-formed expression, concatenate result of the
      expression to input.
    Otherwise, return the input.

    Args:
        s (str): a string, on which calculator should be used

    Returns:
        str

    Hint: safe_eval
    """
    try:
        return ...
    except:
        # expression not well formed! fall back to next token prediction
        return ...


def extract_label(answer: str) -> float:
    try:
        return float(answer.split(">>")[1])
    except:
        return float("nan")
