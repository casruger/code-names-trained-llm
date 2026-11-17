from codenames import Board, generate_board, score_clue_text
from codenames.parsing import parse_clue, parse_guesses
from codenames.prompts import EXAMPLE_CLUE
from codenames.reward import ILLEGAL_CLUE, INVALID_FORMAT, NO_HIT
from codenames.rules import validate_clue


def small_board() -> Board:
    return Board(roles={
        "shark": "team", "whale": "team", "ship": "team",
        "fire": "opponent", "bank": "neutral", "bomb": "assassin",
    })


class FixedGuesser:
    def __init__(self, guesses):
        self.guesses = guesses

    def guess(self, board, clue, number):
        return self.guesses[:number]


def test_board_composition():
    b = generate_board(seed=1)
    assert len(b.words) == 25
    assert len(b.hidden_by_role("team")) == 9
    assert len(b.hidden_by_role("assassin")) == 1
    assert generate_board(seed=1).roles == b.roles  # reproducible


def test_reveal_keeps_assassin_and_team_words():
    b = generate_board(seed=3, reveal_fraction=0.6)
    assert b.revealed
    assert len(b.hidden_by_role("assassin")) == 1
    assert len(b.hidden_by_role("team")) >= 2


def test_rules():
    b = small_board()
    assert validate_clue("ocean", 2, b)[0]
    assert not validate_clue("shark", 1, b)[0]       # on board
    assert not validate_clue("sharks", 1, b)[0]      # plural of board word
    assert not validate_clue("bankrupt", 1, b)[0]    # contains board word
    assert not validate_clue("two words", 1, b)[0]
    assert not validate_clue("ocean", 4, b)[0]       # only 3 team words
    assert not validate_clue(EXAMPLE_CLUE, 2, b)[0]  # copied from the prompt example


def test_parsing():
    assert parse_clue("thinking...\nCLUE: Ocean\nNUMBER: 2") == ("ocean", 2)
    assert parse_clue("**CLUE:** sea\n**NUMBER:** 3") == ("sea", 3)
    assert parse_clue("CLUE: <rabbit>\nNUMBER: 3") == ("rabbit", 3)
    assert parse_clue("CLUE: [board]  \nNUMBER: <4>") == ("board", 4)
    assert parse_clue("I say ocean") is None
    b = small_board()
    assert parse_guesses("GUESSES: Whale, shark, nonsense", b, 3) == ["whale", "shark"]


def test_reward_stops_at_first_wrong_word():
    b = small_board()
    r = score_clue_text(b, "CLUE: ocean\nNUMBER: 3", FixedGuesser(["whale", "fire", "shark"]))
    assert r.outcomes == ["team", "opponent"]
    assert r.reward == 1.0 - 1.0
    r = score_clue_text(b, "CLUE: ocean\nNUMBER: 2", FixedGuesser(["whale", "shark"]))
    assert r.reward == 2.0 and r.n_correct == 2
    r = score_clue_text(b, "CLUE: explode\nNUMBER: 1", FixedGuesser(["bomb"]))
    assert r.reward == -5.0 + NO_HIT
    r = score_clue_text(b, "CLUE: ocean\nNUMBER: 2", FixedGuesser([]))   # guesser found nothing
    assert r.reward == NO_HIT


def test_penalties():
    b = small_board()
    assert score_clue_text(b, "no format", FixedGuesser([])).reward == INVALID_FORMAT
    assert score_clue_text(b, "CLUE: shark\nNUMBER: 1", FixedGuesser([])).reward == ILLEGAL_CLUE


def test_inworld_uses_basic_auth(monkeypatch):
    from codenames.players import LLMClient
    monkeypatch.setenv("INWORLD_API_KEY", "abc123")
    c = LLMClient("some/model", provider="inworld")
    assert str(c.client.base_url).startswith("https://api.inworld.ai/v1")
    assert c.client.default_headers["Authorization"] == "Basic abc123"


def test_reward_function_for_trainer(tmp_path):
    import json as _json
    from codenames.training import CodenamesReward, read_reward_log

    b = small_board()
    calls = []

    class CountingGuesser(FixedGuesser):
        def guess(self, board, clue, number):
            calls.append(clue)
            return super().guess(board, clue, number)

    reward = CodenamesReward(CountingGuesser(["whale", "shark"]), log_path=tmp_path / "log.jsonl")
    board_json = _json.dumps(b.to_dict())
    completions = [  # chat-style completions, as TRL passes them
        [{"role": "assistant", "content": "CLUE: ocean\nNUMBER: 2"}],
        [{"role": "assistant", "content": "CLUE: ocean\nNUMBER: 2"}],   # same clue -> cached
        [{"role": "assistant", "content": "CLUE: shark\nNUMBER: 1"}],   # illegal
        [{"role": "assistant", "content": "no idea"}],                  # unparseable
    ]
    scores = reward(prompts=[None] * 4, completions=completions, board=[board_json] * 4)
    assert scores == [2.0, 2.0, -3.0, -3.0]
    assert calls == ["ocean"]                       # guesser called once thanks to the cache
    log = read_reward_log(tmp_path / "log.jsonl")
    assert len(log) == 4 and log[0]["clue"] == "ocean"
