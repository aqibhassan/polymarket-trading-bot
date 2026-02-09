# /backtest-review

Review backtest results and identify issues.

<instructions>
When the user runs /backtest-review:

1. Read the backtesting skill: `.claude/skills/backtesting/SKILL.md`
2. Ask which strategy and time period to review
3. Check the backtest output for:

RED FLAGS (must fix):
- Sharpe > 3.0 in-sample → likely overfit
- IS Sharpe > 2x OOS Sharpe → likely overfit
- Max drawdown > 20% → risk parameters too loose
- Win rate > 80% → check for lookahead bias
- < 100 trades → insufficient sample size
- Monotonically increasing equity curve → suspicious

YELLOW FLAGS (investigate):
- Profit concentrated in < 5 trades → tail dependency
- Large gap between paper and backtest performance → execution model issue
- Strategy works on only 1 asset → may not generalize

Provide a structured report with:
- Summary verdict: PASS / INVESTIGATE / FAIL
- Specific issues found
- Recommended parameter adjustments
- Next steps
</instructions>
