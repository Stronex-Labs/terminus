# Contributing to Terminus

Thanks for contributing. Terminus is a ruthless backtesting lab — contributions that make it more honest, faster, or more general are welcome.

## What we want

- **New strategy families** — any rule expressible as a vectorized numpy signal
- **Alternative data fetchers** — Kraken, Coinbase, OKX, Bybit spot (implement the `OHLCVFetcher` interface)
- **Portfolio construction methods** — beyond greedy correlation-capped selection
- **New exit methods** — beyond the current 6 exit modes
- **Walk-forward modes** — beyond frozen / anchored-reopt / rolling-reopt
- **Bug reports** — especially where simulation results differ from manual calculation

## What we don't want

- Leverage, shorts, futures, or margin support — Terminus is spot-only by design
- Curve-fit strategies — if it only works on one pair with highly specific parameters, it won't be accepted
- ML models without rigorous walk-forward OOS validation
- Features that add complexity without improving signal quality

## Getting started

```bash
git clone https://github.com/Stronex-Labs/terminus
cd terminus
pip install -e ".[dev]"
pytest
```

## Adding a strategy family

1. Add a signal function in `terminus/strategies.py` — takes OHLCV arrays, returns a boolean entry array
2. Add a config entry in `terminus/configs.py` with the family name and parameter grid
3. Run `terminus sweep --pairs BTCUSDT --tfs 4h` to sanity-check it
4. Open a PR with the sweep results on at least 3 pairs

## Tests

```bash
pytest                      # all tests
pytest tests/test_simulate  # just simulator
```

Coverage must stay above 80%.

## Pull request checklist

- [ ] Tests pass (`pytest`)
- [ ] No hardcoded paths or credentials
- [ ] Strategy tested on at least 3 pairs before claiming generalization
- [ ] Walk-forward results included for new strategies

## License

By contributing you agree your contributions are licensed under MIT.
