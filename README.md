# MarketMind agent tutorial

Build a small financial assistant that can decide when to ask a language model for help and when to retrieve market data from Yahoo Finance.

MarketMind has three parts. The **agent** accepts your question and asks an OpenAI model to choose the next step. **Tools** retrieve a stock price, history, company profile, or financial metrics. **Context** keeps information from earlier turns when the selected implementation supports it.

```text
your question → agent → OpenAI model
                  ↕
              Yahoo Finance tools
                  ↕
          optional conversation context
```

## Before you start

You need Python 3.10 or newer, [uv](https://docs.astral.sh/uv/), and an OpenAI API key. The examples make live requests to OpenAI and Yahoo Finance; usage charges and data availability apply.

## Install

```bash
git clone https://github.com/agenteer/market-mind-agent-tutorial.git
cd market-mind-agent-tutorial
uv venv
source .venv/bin/activate
uv pip install -e .
cp .env.example .env
```

Open `.env`, replace `your_api_key_here` with your OpenAI API key, and save the file. `.env` is ignored by Git.

Check that the command is available:

```bash
market-mind --help
```

## Run an implementation

Start with the OpenAI Agent SDK version. Ask a question such as `What is the latest price of AAPL?`, then type `exit` when you are done.

```bash
market-mind openai-agent-sdk
```

The repository also includes two implementations built from API calls:

```bash
market-mind chat-completion
market-mind response-api
```

Add `--debug` to any command to write diagnostic logs to `logs/`.

## Explore the code

- `src/common/tools_yf.py` contains the Yahoo Finance tools.
- `src/openai_agent_sdk/` uses the OpenAI Agents SDK and can preserve conversation context.
- `src/agent_from_scratch/` shows the Chat Completions and Responses API approaches.
- `src/cli/main.py` connects each implementation to the `market-mind` command.

## Tutorial and license

Read the accompanying [MarketMind tutorial](https://agenteer.com/learn/tutorials/market-mind-agent/). This project is released under the [MIT License](LICENSE).
