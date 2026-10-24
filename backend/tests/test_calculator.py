from services.calculator import calculate


def test_line_items_and_total():
    r = calculate(expressions=["4*750", "2,300", "₦1,500", "2.5k"])
    assert r["total"] == "9300"
    assert r["total_formatted"] == "9,300"
    assert [i["value"] for i in r["results"]] == ["3000", "2300", "1500", "2500"]


def test_decimal_exactness():
    assert calculate(expression="0.1+0.2")["result"] == "0.3"


def test_function_args_not_treated_as_thousands():
    assert calculate(expression="max(10,100)")["result"] == "100"


def test_rejects_unsafe_input():
    for bad in ['__import__("os").system("x")', "open('f')", "9**9**9", "a+1", "'x'*5"]:
        assert "error" in calculate(expression=bad), bad


def test_errors_are_actionable():
    assert "error" in calculate(expression="15%*20000")
    assert "error" in calculate(expressions=["1+1", "oops"])
    assert "error" in calculate()
