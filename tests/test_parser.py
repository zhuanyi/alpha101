import pytest

from alpha101 import load_alphas
from alpha101.expr.ast import Binary, Call, Num, Ternary, Unary, Var
from alpha101.expr.parser import ParseError, parse

ALPHAS = load_alphas()


@pytest.mark.parametrize("alpha_id", sorted(ALPHAS))
def test_every_paper_alpha_parses_and_round_trips(alpha_id):
    node = parse(ALPHAS[alpha_id].expression)
    canon = node.to_str()
    again = parse(canon)
    assert again == node
    assert again.to_str() == canon


def test_left_associative_minus_and_divide():
    assert parse("open - high - low") == \
        Binary("-", Binary("-", Var("open"), Var("high")), Var("low"))
    assert parse("open / high / low") == Binary("/", Binary("/", Var("open"), Var("high")), Var("low"))


def test_power_binds_tighter_than_unary_minus_and_is_right_associative():
    assert parse("-close ^ 2") == Unary("-", Binary("^", Var("close"), Num(2)))
    assert parse("2 ^ 3 ^ 2") == Binary("^", Num(2), Binary("^", Num(3), Num(2)))
    assert parse("close ^ -1") == Binary("^", Var("close"), Num(-1))


def test_multiplication_over_addition_and_comparison_over_or():
    assert parse("open + high * low") == Binary("+", Var("open"), Binary("*", Var("high"), Var("low")))
    n = parse("open < high || low == close")
    assert n == Binary("||", Binary("<", Var("open"), Var("high")), Binary("==", Var("low"), Var("close")))


def test_ternary_is_lowest_and_right_associative():
    n = parse("open < 1 ? high : low < 2 ? close : volume")
    assert n == Ternary(Binary("<", Var("open"), Num(1)), Var("high"),
                        Ternary(Binary("<", Var("low"), Num(2)), Var("close"), Var("volume")))


def test_case_insensitive_and_aliases():
    assert parse("Ts_ArgMax(close, 5)") == Call("ts_argmax", (Var("close"), Num(5)))
    assert parse("SignedPower(close, 2.)") == Call("signedpower", (Var("close"), Num(2)))
    assert parse("IndNeutralize(close, IndClass.sector)").to_str() == "indneutralize(close, indclass.sector)"


def test_min_max_resolution():
    assert parse("min(close, 5)").name == "ts_min"
    assert parse("max(close, open)").name == "elementwise_max"


def test_numbers():
    assert parse(".001") == Num(0.001)
    assert parse("2.") == Num(2.0)
    assert parse("(-1 * 1)") == Binary("*", Num(-1), Num(1))


def test_adv_fields():
    assert parse("adv180") == Var("adv180")


@pytest.mark.parametrize("bad", ["close +", "foo(close)", "rank(close", "close close",
                                 "ts_rank(close)", "IndClass.galaxy", "price", "close $ 2"])
def test_malformed_input_raises(bad):
    with pytest.raises(ParseError):
        parse(bad)
