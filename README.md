# code-names-trained-llm
This is a code names spymaster LLM trained by myself as a learnings project.

Doel: een klein open LLM (Qwen) met reinforcement learning (GRPO) trainen tot een
goede Codenames-**spymaster**: één woord + getal dat zoveel mogelijk eigen woorden
combineert zonder de tegenstander, neutrale woorden of de moordenaar te raken.

## Hoe een episode werkt

```
bord (25 woorden) ──► spymaster (model dat we trainen) ──► "CLUE: ocean / NUMBER: 3"
                                                              │
        reward ◄── score_guesses ◄── guesser (vast model of jij)
```

| Uitkomst | Reward |
|---|---|
| eigen woord geraden | +1 per woord |
| neutraal woord | −0,5 (beurt stopt) |
| woord van tegenstander | −1 (beurt stopt) |
| moordenaar | −5 (beurt stopt) |
| geen enkel eigen woord geraden (nutteloze hint) | −1 extra |
| onleesbaar antwoord / illegale hint (op bord, deelwoord, voorbeeld uit prompt gekopieerd) | −3 |

Pas de waarden aan in `codenames/reward.py` en kijk hoe de strategie verandert.

## Structuur

```
codenames/
  words.py     woordenlijst (~400 Engelse woorden van min. 5 letters, zelf samengesteld)
  board.py     bord genereren (9 team / 8 tegenstander / 7 neutraal / 1 moordenaar)
  rules.py     is de hint legaal? (niet op bord, geen deel-/vervoegd woord)
  prompts.py   prompts voor spymaster en guesser (zelfde in baseline én training)
  parsing.py   modeltekst -> hint / gokken
  reward.py    reward-functie
  players.py   LLM- (OpenAI-compatibel), mens- en random-spelers
  game.py      score_clue_text(): de reward-functie die de training gebruikt
baseline.py    N borden spelen en gemiddelde score meten
tests/         pytest
```

## Lokaal draaien

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
pytest                            # alles groen?

# zonder LLM: jij bent spymaster, random guesser
python baseline.py --spymaster human --guesser random --n 3 --runs 1
```

### Baseline meten

Spymaster = Qwen lokaal via Ollama, guesser = **Llama 3.3 70B via Inworld** (vast
voor baseline, training én eindtest — nooit wisselen, anders zijn scores niet vergelijkbaar).

1. Installeer [Ollama](https://ollama.com) en `ollama pull qwen2.5:1.5b`
2. Kopieer `.env.example` naar `.env` en vul je Inworld key in (`.env` staat in
   `.gitignore` en komt dus nooit op GitHub)
3. Kopieer het exacte model-ID van Llama 3.3 70B uit het Inworld-portal en draai
   5 runs op dezelfde 50 evaluatieborden:

```bash
python baseline.py --model qwen2.5:1.5b --guesser-provider inworld --guesser-model <inworld-llama-id> --runs 5
```

Tip: test eerst met `--runs 1 --n 2` of de API-key en het model-ID kloppen.

Resultaten komen in `eval_results/<spymaster>__vs__<guesser>/`:

| Bestand | Inhoud |
|---|---|
| `runs.csv` | per run: mean reward, % geldige hints, gem. getal, gem. goed, % moordenaar/tegenstander/neutraal |
| `summary.json` | dezelfde metrics + mean / std / min / max over de runs |
| `run_k.jsonl` | per bord: bord, ruwe modeloutput, hint, gokken, uitkomst |

De evaluatieborden (seed vanaf `EVAL_SEED` in `codenames/board.py`) zijn elke run
hetzelfde en worden nooit gebruikt voor training. Verschil tussen runs = toeval van
het model; verschil tussen modellen = wat de training heeft opgeleverd.

Na de training, zelfde borden en zelfde guesser:

```bash
python baseline.py --model codenames-spymaster --guesser-provider inworld --guesser-model <inworld-llama-id> --runs 5
```

Zelf raden tegen de LLM-spymaster: `python baseline.py --model qwen2.5:1.5b --guesser human --n 5 --runs 1`

Providers: `ollama` (lokaal), `inworld` (`INWORLD_API_KEY`), `openrouter`
(`OPENROUTER_API_KEY`), `openai` (`OPENAI_API_KEY`) via `--provider` / `--guesser-provider`.
De ruwe guesser-output staat per bord in `run_k.jsonl` (`guesser_raw`).

## Volgende stap

`train_grpo.ipynb` voor Colab: Qwen + LoRA trainen met TRL's GRPOTrainer, met
`score_clue_text` als reward-functie.
