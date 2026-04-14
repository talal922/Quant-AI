# trading_env.py

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ForexTradingEnv(gym.Env):
    """
    RL Stock Trading Environment — AAPL-optimized.

    Reward function redesign vs original:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ Component            │ Old                  │ New                        │
    ├──────────────────────┼──────────────────────┼────────────────────────────┤
    │ Base                 │ log(equity/prev)     │ unchanged                  │
    │ Drawdown penalty     │ linear on Δ DD only  │ quadratic on |DD| level    │
    │                      │ −Δdd·penalty·0.5     │ −Δdd·penalty·0.3           │
    │                      │ (no persist. cost)   │ −dd²·penalty·7.5 (persist) │
    │ Unrealized bonus     │ flat +0.0005         │ tanh(unreal·5)·0.003       │
    │ Holding loser        │ none                 │ −|unreal|·log(t+1)·0.0005  │
    │ Realized close       │ 0.0                  │ tanh(pct·3)·0.02 for wins  │
    │                      │                      │ pct·0.3 for losses (pct<0) │
    └──────────────────────┴──────────────────────┴────────────────────────────┘

    The quadratic drawdown term is anchored so that:
      DD=2%  → 0.30× old penalty (lenient — normal trading noise)
      DD=10% → 1.50× old penalty (significantly more costly)
      DD=15% → 2.25× old penalty (severely punished)
      DD=20% → 3.00× old penalty (catastrophic — agent must avoid at all costs)

    This profile is tuned for AAPL, where a 20% drawdown from peak would represent
    wiping out a full quarter's institutional gains — an unacceptable outcome.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 14,
        feature_columns=None,
        position_size_pct: float = 0.95,
        commission_pct: float = 0.001,
        allow_short: bool = False,
        random_start: bool = True,
        min_episode_steps: int = 300,
        episode_max_steps: int | None = 252,
        obs_noise_std: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.feature_columns = feature_columns if feature_columns else list(self.df.columns)
        self.window_size = int(window_size)

        self.position_size_pct = float(position_size_pct)
        self.commission_pct = float(commission_pct)
        self.allow_short = bool(allow_short)
        self.drawdown_penalty = float(kwargs.get("drawdown_penalty", 1.5))

        self.random_start = random_start
        self.min_episode_steps = min_episode_steps
        self.episode_max_steps = episode_max_steps  # default 252 = 1 trading year
        # obs_noise_std: Gaussian noise added to features during training only.
        # Acts as a data-augmentation regularizer — the agent can't memorize exact
        # price sequences. Set to 0.0 at test time (no noise during backtest).
        self.obs_noise_std = float(obs_noise_std)

        # Simplified actions: Hold, Buy, Close
        self.action_space = spaces.Discrete(3)

        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 3
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32,
        )
        self._reset_state()

    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.shares_held = 0.0

        self.initial_equity_usd = 10_000.0
        self.equity_usd = self.initial_equity_usd
        self.cash_usd = self.initial_equity_usd
        self.peak_equity = self.initial_equity_usd
        self.max_drawdown_pct = 0.0
        self.prev_drawdown = 0.0

        self.equity_curve = []
        self.last_trade_info = None
        self.max_price_since_open = 0.0

    # ── Observation helpers ──────────────────────────────────────────────────

    def _get_state_features(self):
        pos = float(self.position)
        t_norm = np.tanh(self.time_in_trade / 20.0)
        unreal_pct = 0.0
        if self.position != 0 and self.entry_price is not None:
            # FIX: Use max(0, current_step - 1) to prevent Data Leak
            safe_step = max(0, self.current_step - 1)
            close_price = float(self.df.loc[safe_step, "Close"])
            
            if self.position == 1:
                unreal_pct = (close_price - self.entry_price) / self.entry_price
            else:
                unreal_pct = (self.entry_price - close_price) / self.entry_price
                
        unreal_pct = np.tanh(unreal_pct * 10.0)
        return np.array([pos, t_norm, unreal_pct], dtype=np.float32)

    def _get_observation(self):
        start = max(0, self.current_step - self.window_size)
        obs_df = self.df.iloc[start : self.current_step][self.feature_columns]

        if len(obs_df) == 0:
            base = np.tile(
                self.df.iloc[0][self.feature_columns].values.astype(np.float32),
                (self.window_size, 1),
            )
        else:
            base = obs_df.values.astype(np.float32)
            if base.shape[0] < self.window_size:
                pad_rows = self.window_size - base.shape[0]
                base = np.vstack([np.tile(base[0], (pad_rows, 1)), base])

        state_feat = self._get_state_features()
        state_block = np.tile(state_feat, (self.window_size, 1))
        obs = np.hstack([base, state_block]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        # Observation noise for regularization (only active when obs_noise_std > 0).
        # Prevents the agent from memorizing exact feature sequences in training data.
        # Applied only to the feature columns (first base_num_features columns),
        # NOT to the portfolio state columns (position, time_in_trade, unrealized_pct).
        if self.obs_noise_std > 0.0:
            noise = np.random.normal(0.0, self.obs_noise_std,
                                     size=(self.window_size, self.base_num_features)).astype(np.float32)
            obs[:, :self.base_num_features] += noise

        return obs

    # ── Trade mechanics ──────────────────────────────────────────────────────

    def _open_position(self, direction: int = 1):
        """
        Open a position with ATR-based dynamic SL/TP.

        WHY DYNAMIC SL/TP:
        Fixed 1.5%/3.0% SL/TP ignores market volatility. During high-volatility
        periods (AAPL earnings, macro events), normal price noise easily exceeds
        1.5% → constant SL hits with no signal. During calm periods, 3% TP is
        too far to ever be reached → the agent holds forever.

        ATR-based SL/TP adapts to the current regime:
          - Low vol  (ATR=0.5%): SL=0.75%, TP=1.50%  — tight, quick trades
          - Normal   (ATR=1.5%): SL=2.25%, TP=4.50%  — standard
          - High vol (ATR=3.0%): SL=3.00%, TP=6.00%  — wide, fewer false SL hits
        Clamped: SL ∈ [0.8%, 4.0%], TP = SL × 2.0 (constant 1:2 R:R).
        """
        entry = float(self.df.loc[self.current_step, "Close"])
        self.max_price_since_open = entry

        # ATR-based SL: 1.5× current ATR percentage, clamped to safe range
        if "atr_14" in self.df.columns:
            atr_pct = float(self.df.loc[self.current_step, "atr_14"]) / (entry + 1e-9) * 100.0
        else:
            atr_pct = 1.5
        sl_pct = float(np.clip(atr_pct * 1.5, 0.8, 4.0))
        tp_pct = sl_pct * 2.0  # constant 1:2 R:R

        invest_amount = self.equity_usd * self.position_size_pct
        commission_cost = invest_amount * self.commission_pct

        self.shares_held = (invest_amount - commission_cost) / entry
        self.cash_usd -= invest_amount

        if direction == 1:
            self.sl_price = entry * (1 - sl_pct / 100.0)
            self.tp_price = entry * (1 + tp_pct / 100.0)
            self.position = 1
        else:
            self.sl_price = entry * (1 + sl_pct / 100.0)
            self.tp_price = entry * (1 - tp_pct / 100.0)
            self.position = -1

        self.entry_price = entry
        self.time_in_trade = 0

        self.last_trade_info = {
            "event": "OPEN",
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
        }

    def _close_position(self, reason: str, exit_price: float) -> float:
        """
        Close the current position and return a shaping reward component.

        Returns a float (was always 0.0 in the original).  The shaping value is:
          • Profitable close:  tanh(profit_pct × 3) × 0.02
            At +1%: +0.0006,  +3%: +0.0018,  +5%: +0.0030,  +10%: +0.0039
          • Meaningful loss (< −0.5%):  profit_pct × 0.3   (profit_pct is negative)
            At −1%: −0.003,  −3%: −0.009  — extra sting beyond the equity change
            already captured in the log-return, to reinforce cutting losses faster.

        Why asymmetric?  AAPL's SL-induced drawdowns compound quickly on trend
        reversals.  Making realized losses "hurt twice" discourages the agent from
        accepting marginal entries.
        """
        position_value = self.shares_held * exit_price
        if self.position == -1:
            pnl = (self.entry_price - exit_price) * self.shares_held
            position_value = (self.shares_held * self.entry_price) + pnl

        commission_cost = position_value * self.commission_pct
        net_proceeds = position_value - commission_cost

        gross_price_return = (
            (exit_price - self.entry_price) / self.entry_price
            if self.position == 1
            else (self.entry_price - exit_price) / self.entry_price
        )
        profit_pct = float(gross_price_return - 2.0 * self.commission_pct)

        self.cash_usd += net_proceeds
        self.equity_usd = self.cash_usd
        net_profit_usd = position_value - (self.shares_held * self.entry_price) - commission_cost

        trade_info = {
            "event": "CLOSE",
            "reason": reason,
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "net_pips": float(net_profit_usd),
            "profit_pct": profit_pct,
            "time_in_trade": int(self.time_in_trade),
        }

        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.shares_held = 0.0
        self.time_in_trade = 0
        self.max_price_since_open = 0.0
        self.last_trade_info = trade_info

        # ── Realized trade shaping ──────────────────────────────────────────
        if profit_pct > 0.0:
            # Extra reward for profitable close — reinforces locking in AAPL gains
            # before reversal.  tanh bounds it: no arbitrarily large shaping signals.
            shaping_reward = np.tanh(profit_pct * 3.0) * 0.02
        elif profit_pct < -0.005:
            # Meaningful realized loss: discourages accepting bad entries.
            # profit_pct is already negative, so this subtracts from reward.
            shaping_reward = profit_pct * 0.3
        else:
            shaping_reward = 0.0

        return float(shaping_reward)

    def _check_sl_tp_intrabar_and_maybe_close(self) -> float | None:
        if self.position == 0:
            return None

        current_close = float(self.df.loc[self.current_step, "Close"])

        # Trailing stop (long only) — logic unchanged
        if self.position == 1:
            if current_close > self.max_price_since_open:
                self.max_price_since_open = current_close
                new_sl = self.max_price_since_open * 0.98
                if new_sl > self.sl_price:
                    self.sl_price = new_sl

        if self.current_step >= self.n_steps - 2:
            return self._close_position("END_OF_DATA", current_close)

        next_high = float(self.df.loc[self.current_step + 1, "High"])
        next_low  = float(self.df.loc[self.current_step + 1, "Low"])

        if self.position == 1:
            if next_low <= self.sl_price and next_high >= self.tp_price:
                return self._close_position("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
            elif next_low <= self.sl_price:
                return self._close_position("SL_HIT", self.sl_price)
            elif next_high >= self.tp_price:
                return self._close_position("TP_HIT", self.tp_price)
        else:
            if next_high >= self.sl_price and next_low <= self.tp_price:
                return self._close_position("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
            elif next_high >= self.sl_price:
                return self._close_position("SL_HIT", self.sl_price)
            elif next_low <= self.tp_price:
                return self._close_position("TP_HIT", self.tp_price)

        return None

    # ── Gymnasium interface ──────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        if self.random_start:
            max_start = self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            self.current_step = int(
                np.random.randint(self.window_size, max(self.window_size + 1, max_start))
            )
        else:
            self.current_step = self.window_size
        return self._get_observation(), {}

    def step(self, action: int):
        if self.terminated or self.truncated:
            return self._get_observation(), 0.0, True, False, {}

        self.steps_in_episode += 1
        prev_equity = self.equity_usd
        prev_drawdown = self.prev_drawdown

        # BUG FIX: Reset last_trade_info at the start of EVERY step.
        # Previously it was never cleared, so the same OPEN/CLOSE event was
        # emitted on every subsequent step until the next real trade occurred.
        # This caused rows 4,5,6 (same OPEN at step 65) and rows 7,8,9 (same
        # CLOSE at step 68) in the trade history — phantom duplicates that
        # inflated loss count and distorted all trade metrics.
        self.last_trade_info = None

        # ── Execute action ───────────────────────────────────────────────────
        trade_shaping = 0.0
        if action == 1 and self.position == 0:
            self._open_position(1)
        elif action == 2 and self.position != 0:
            trade_shaping += self._close_position(
                "MANUAL_CLOSE", float(self.df.loc[self.current_step, "Close"])
            )

        sl_tp_result = self._check_sl_tp_intrabar_and_maybe_close()
        if sl_tp_result is not None:
            trade_shaping += sl_tp_result

        # ── Advance step ─────────────────────────────────────────────────────
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            self.terminated = True
        if self.episode_max_steps and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        # ── Update equity ────────────────────────────────────────────────────
        if self.position != 0:
            self.time_in_trade += 1
            current_close = float(self.df.loc[self.current_step, "Close"])
            pos_val = self.shares_held * current_close
            self.equity_usd = self.cash_usd + pos_val
        else:
            self.equity_usd = self.cash_usd

        if self.equity_usd > self.peak_equity:
            self.peak_equity = self.equity_usd

        drawdown = max(
            0.0, (self.peak_equity - self.equity_usd) / max(self.peak_equity, 1e-9)
        )
        drawdown_increase = max(0.0, drawdown - prev_drawdown)
        self.max_drawdown_pct = max(self.max_drawdown_pct, drawdown)

        # ── Reward computation ────────────────────────────────────────────────
        # 1. Primary signal: log return (unchanged, mathematically correct)
        log_return = np.log(self.equity_usd / max(prev_equity, 1e-9) + 1e-9)
        reward = log_return

        # 2. Inactivity penalty: break the "never trade" bias.
        #    Holding cash gives a tiny nudge to explore trading.
        #    Small enough to not override real signals, big enough to matter.
        if self.position == 0:
            reward -= 0.0001

        # 3. Drawdown shaping — asymmetric quadratic (FIXED: was 7.5, now 0.5)
        #
        #    a) Delta penalty: penalises each bar that makes the drawdown WORSE.
        reward -= drawdown_increase * self.drawdown_penalty * 0.3
        #
        #    b) Absolute level penalty: penalises SUSTAINING a drawdown.
        #       FIX: coefficient reduced from 7.5 → 0.5 to prevent the penalty
        #       from completely dominating the log-return signal.
        #       At 7.5 a 5% DD gives -0.075/step vs typical log-return of 0.001-0.003
        #       — the agent learns "never trade" because it's the only safe policy.
        #       At 0.5 the relationship is balanced:
        #       DD= 2% → 0.0003  (reasonable cost)
        #       DD=10% → 0.0075  (meaningful cost)
        #       DD=20% → 0.0300  (severe but not suffocating)
        reward -= (drawdown ** 2) * self.drawdown_penalty * 0.5

        # 4. Unrealized P&L shaping
        #    Calculated AFTER current_step increment so it reflects current mark-to-mkt.
        if self.position != 0 and self.entry_price is not None:
            current_close = float(self.df.loc[self.current_step, "Close"])
            unrealized_profit_pct = (
                (current_close - self.entry_price) / self.entry_price
                if self.position == 1
                else (self.entry_price - current_close) / self.entry_price
            )

            if unrealized_profit_pct > 0.0:
                # Scaled bonus — proportional to unrealized profit, bounded by tanh.
                # Old flat +0.0005 gave the same bonus whether +0.1% or +10% in profit.
                # New:  at +1%  → +0.00015
                #       at +5%  → +0.00073
                #       at +10% → +0.00139
                reward += np.tanh(unrealized_profit_pct * 5.0) * 0.003
            else:
                # Time-weighted loser penalty: the longer the agent holds a losing
                # AAPL position, the more expensive it becomes.  Encourages quick
                # loss-cutting rather than hoping for mean-reversion.
                # At −2% for 1 bar:  0.02·log(2)·0.0005 ≈ 0.000007 (tiny)
                # At −2% for 20 bars: 0.02·log(21)·0.0005 ≈ 0.000030 (noticeable)
                # At −5% for 30 bars: 0.05·log(31)·0.0005 ≈ 0.000086 (meaningful)
                reward -= (
                    abs(unrealized_profit_pct) * np.log1p(self.time_in_trade) * 0.0005
                )

        # 5. Realized trade shaping (from _close_position)
        reward += trade_shaping

        self.prev_drawdown = drawdown
        self.equity_curve.append(float(self.equity_usd))

        info = {
            "equity_usd": float(self.equity_usd),
            "position": int(self.position),
            "last_trade_info": self.last_trade_info,
            "max_drawdown_pct": float(self.max_drawdown_pct),
        }

        return self._get_observation(), float(reward), self.terminated, self.truncated, info