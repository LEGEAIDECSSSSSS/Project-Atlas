from services.listparse import parse_line, try_total_list
from tests.conftest import sse

MARKET = """Meat- 40k 8 kg
Noodles-19k one carton
Macaroni-4k 4 packs
Tomatoes-8k one rubber
Pepper- 8k one rubber
Spaghetti-8k 8 packs
Tomatoe paste-3k 3 rows
Onions-3k half rubber
Egg-6k one crate
Milk- 6500 2 cowbell packs ozzy
Cameroon pepper 1k jakande ( check market if you will see 1k own if not go to roundabout market )
Knorr-3k
Curry ( refill scahet)
Dish soap-3k ozzy
(Adobo spice- 7k ebeano )
Bay leaf-1k
Ginger mix-1k (check market or ebeano if not buy from supersaver)
Detergent- 2k pink kiln or good mama (ebeano or ozzy)
calculate the total"""


def test_real_market_list_total():
    r = try_total_list(MARKET)
    assert r["total"] == "123500" and r["items"] == 17
    assert "**No price given, so not counted:** Curry." in r["text"]
    assert "calculate the total" not in r["text"]          # instruction line is not an item
    assert "320" not in r["text"]                          # 8 kg is not multiplied by 40k


def test_line_styles():
    amount = lambda s: parse_line(s)["amount"]
    assert amount("Meat- 40k 8 kg") == 40000
    assert amount("Milk- 6500 2 packs") == 6500
    assert amount("Rice ₦1,500") == 1500
    assert amount("Fish 2.5k") == 2500
    assert amount("Eggs 12") is None                       # a small bare number is a quantity
    assert amount("Soap 4 x 750") == 3000                  # explicit multiplication
    assert amount("Sugar 2 packs 500 each") == 1000
    assert parse_line("Market list:") is None
    assert parse_line("please add it up for me") is None


def test_not_triggered_for_other_messages():
    assert try_total_list("what is the total population of Lagos?") is None
    assert try_total_list("Rice 5k\nBeans 3k\nadd them up") is None   # only 2 priced lines
    assert try_total_list("Rice 5k\nBeans 3k\nYam 2k\nWrite a poem about food") is None  # no total requested
    assert try_total_list("```\nRice 5k\nBeans 3k\nYam 2k\n```\ncalculate the total") is None


def test_currency_symbol_reused():
    r = try_total_list("Rice ₦5,000\nBeans ₦3,000\nYam ₦2,000\ntotal?")
    assert "₦10,000" in r["text"]


def test_chat_endpoint_answers_without_the_model(client, script):
    script.installed = []                                  # even with no models installed
    events = sse(client, {"message": MARKET, "conversation_id": "list-conversation-1"})
    assert not [e for e in events if e["type"] == "error"]
    assert next(e for e in events if e["type"] == "meta")["model"] == "built-in calculator"
    assert "Grand total: 123,500" in "".join(e["text"] for e in events if e["type"] == "token")
    assert script.requests == []                           # the model was never called
    saved = client.get("/conversations/list-conversation-1").json()["messages"]
    assert saved[-1]["meta"]["tools"] == ["Parsed the list and added it up exactly"]
