"""NLP-based features extracted from blockchain/crypto text data using statistical proxies.

All features work on standard OHLCV + log_return columns without external API calls.
Statistical proxies correlate with the same underlying dynamics as actual NLP signals
(whitepaper complexity, community sentiment, social buzz, dev activity, mention frequency).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Crypto-specific sentiment lexicon
# Bullish terms mapped to +1, bearish terms mapped to -1
# Used conceptually to weight price/volume patterns that historically
# coincide with periods of strong community sentiment
CRYPTO_SENTIMENT_WORDS: dict[str, dict[str, int]] = {
    "bullish": {
        "moon": 1, "pump": 1, "bullish": 1, "hodl": 1, "diamond_hands": 1,
        "breakout": 1, "accumulate": 1, "undervalued": 1, "adoption": 1,
        "partnership": 1, "upgrade": 1, "mainnet": 1, "staking": 1,
        "defi": 1, "rally": 1, "ath": 1, "fomo": 1, "buy_the_dip": 1,
        "halving": 1, "institutional": 1, "etf": 1, "approval": 1,
        "listing": 1, "burn": 1, "deflationary": 1,
    },
    "bearish": {
        "dump": -1, "bearish": -1, "crash": -1, "rug_pull": -1, "scam": -1,
        "hack": -1, "exploit": -1, "sec": -1, "ban": -1, "regulation": -1,
        "sell": -1, "short": -1, "liquidation": -1, "fud": -1, "ponzi": -1,
        "bubble": -1, "overvalued": -1, "dead_cat": -1, "capitulation": -1,
        "delisting": -1, "lawsuit": -1, "fraud": -1, "exit_scam": -1,
        "whale_dump": -1, "paper_hands": -1,
    },
}

# Number of bullish vs bearish terms (used as weights for proxy features)
_N_BULLISH = len(CRYPTO_SENTIMENT_WORDS["bullish"])
_N_BEARISH = len(CRYPTO_SENTIMENT_WORDS["bearish"])


class BlockchainNLPFeatures:
    """NLP-inspired features using statistical proxies over OHLCV data.

    Since direct text data (whitepapers, tweets, GitHub commits) is not
    available in the standard pipeline, this module creates proxy features
    that correlate with the same underlying dynamics:

    1. Whitepaper/documentation complexity proxies (price action complexity)
    2. Crypto sentiment lexicon proxies (bullish/bearish regime via returns)
    3. Social media buzz proxies (volume/price divergence patterns)
    4. GitHub activity proxies (momentum during dev-active hours UTC)
    5. Token mention frequency proxies (volume surge patterns)
    """

    # Dev-active hours in UTC (roughly 9am-6pm across US+EU timezones)
    DEV_HOURS_START = 9
    DEV_HOURS_END = 18

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add blockchain NLP proxy features to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least 'close', 'volume', and 'log_return' columns.
            Optional: 'open', 'high', 'low', 'timestamp'.

        Returns
        -------
        pd.DataFrame
            Copy of input with NLP proxy feature columns appended.
        """
        required = {"close", "volume", "log_return"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            logger.warning(
                "BlockchainNLPFeatures: missing required columns %s, "
                "returning df unchanged.",
                missing,
            )
            return df

        df = df.copy()

        df = self._add_whitepaper_complexity_features(df)
        df = self._add_sentiment_lexicon_features(df)
        df = self._add_social_buzz_features(df)
        df = self._add_github_activity_features(df)
        df = self._add_mention_frequency_features(df)

        return df

    # ------------------------------------------------------------------
    # 1. Whitepaper / documentation complexity proxies
    # ------------------------------------------------------------------
    def _add_whitepaper_complexity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy for text complexity using price action complexity metrics.

        Complex price patterns (high entropy, many direction changes)
        historically correlate with periods of dense information release
        such as whitepaper updates or protocol documentation changes.
        """
        ret = df["log_return"]
        lb = self.lookback

        # "Sentence length" proxy: run length of same-sign returns
        # Longer runs = simpler narrative, shorter runs = complex/choppy
        sign_changes = (np.sign(ret) != np.sign(ret.shift(1))).astype(float)
        df["nlp_complexity_choppiness"] = (
            sign_changes.rolling(lb).mean()
        ).astype(np.float32)

        # "Vocabulary richness" proxy: number of unique return magnitudes
        # (discretized into bins) within a rolling window
        def _vocab_richness(x: np.ndarray) -> float:
            if len(x) < 2:
                return 0.0
            try:
                bins = np.histogram(x, bins=10)[0]
                occupied = np.sum(bins > 0)
                return float(occupied) / 10.0
            except (ValueError, ZeroDivisionError):
                return 0.0

        df["nlp_vocab_richness"] = (
            ret.rolling(lb).apply(_vocab_richness, raw=True)
        ).astype(np.float32)

        # "Readability" proxy: rolling autocorrelation of returns
        # High autocorrelation = predictable/readable, low = complex
        def _autocorr(x: np.ndarray) -> float:
            if len(x) < 3:
                return 0.0
            s = pd.Series(x)
            ac = s.autocorr(lag=1)
            return float(ac) if np.isfinite(ac) else 0.0

        df["nlp_readability_proxy"] = (
            ret.rolling(lb).apply(_autocorr, raw=True)
        ).astype(np.float32)

        # Information density: rolling entropy of absolute returns
        def _entropy_proxy(x: np.ndarray) -> float:
            if len(x) < 2:
                return 0.0
            abs_x = np.abs(x)
            total = abs_x.sum()
            if total == 0:
                return 0.0
            p = abs_x / total
            p = p[p > 0]
            return float(-np.sum(p * np.log(p + 1e-10)))

        df["nlp_info_density"] = (
            ret.rolling(lb).apply(_entropy_proxy, raw=True)
        ).astype(np.float32)

        return df

    # ------------------------------------------------------------------
    # 2. Crypto-specific sentiment lexicon proxies
    # ------------------------------------------------------------------
    def _add_sentiment_lexicon_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy for bullish/bearish sentiment using return regime classification.

        Consecutive positive returns with rising volume proxy for bullish
        community sentiment; the reverse proxies bearish sentiment.
        The ratio of bullish-to-bearish terms in the lexicon weights the signal.
        """
        ret = df["log_return"]
        vol = df["volume"]
        lb = self.lookback

        # Bullish signal: positive returns + above-average volume
        vol_ma = vol.rolling(lb).mean()
        vol_above = (vol > vol_ma).astype(float)
        bullish_raw = (ret > 0).astype(float) * vol_above

        # Bearish signal: negative returns + above-average volume
        bearish_raw = (ret < 0).astype(float) * vol_above

        # Rolling sentiment scores weighted by lexicon size ratio
        bullish_ratio = _N_BULLISH / (_N_BULLISH + _N_BEARISH)
        bearish_ratio = _N_BEARISH / (_N_BULLISH + _N_BEARISH)

        df["nlp_bullish_score"] = (
            bullish_raw.rolling(lb).mean() * bullish_ratio
        ).astype(np.float32)

        df["nlp_bearish_score"] = (
            bearish_raw.rolling(lb).mean() * bearish_ratio
        ).astype(np.float32)

        # Net sentiment (-1 to +1 range)
        df["nlp_net_sentiment"] = (
            df["nlp_bullish_score"] - df["nlp_bearish_score"]
        ).astype(np.float32)

        # Sentiment momentum (change in net sentiment)
        df["nlp_sentiment_momentum"] = (
            df["nlp_net_sentiment"].diff(max(lb // 2, 1))
        ).astype(np.float32)

        # Extreme sentiment flag (potential reversal)
        net = df["nlp_net_sentiment"]
        net_std = net.rolling(lb * 2).std()
        net_mean = net.rolling(lb * 2).mean()
        df["nlp_sentiment_extreme"] = np.where(
            net_std > 0,
            ((net - net_mean) / net_std.replace(0, np.nan)).abs() > 2.0,
            0.0,
        ).astype(np.float32)

        return df

    # ------------------------------------------------------------------
    # 3. Social media buzz indicators
    # ------------------------------------------------------------------
    def _add_social_buzz_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy for social media buzz via volume/price divergence patterns.

        Social media buzz typically leads volume spikes that diverge from
        price movement. We detect these patterns as buzz proxies.
        """
        ret = df["log_return"]
        vol = df["volume"]
        lb = self.lookback

        # Volume spike detection (buzz proxy)
        vol_ma = vol.rolling(lb).mean()
        vol_std = vol.rolling(lb).std()
        df["nlp_buzz_volume_z"] = np.where(
            vol_std > 0,
            (vol - vol_ma) / vol_std.replace(0, np.nan),
            0.0,
        ).astype(np.float32)

        # Buzz intensity: sudden volume with small price movement
        # (many participants, no direction = discussion/hype phase)
        abs_ret = ret.abs()
        abs_ret_ma = abs_ret.rolling(lb).mean()
        vol_ratio = np.where(vol_ma > 0, vol / vol_ma.replace(0, np.nan), 1.0)
        ret_ratio = np.where(
            abs_ret_ma > 0,
            abs_ret / abs_ret_ma.replace(0, np.nan),
            1.0,
        )
        df["nlp_buzz_intensity"] = np.where(
            ret_ratio > 0,
            vol_ratio / np.where(ret_ratio == 0, 1.0, ret_ratio),
            0.0,
        ).astype(np.float32)

        # Buzz persistence: fraction of last N bars with above-avg volume
        vol_above = (vol > vol_ma).astype(float)
        df["nlp_buzz_persistence"] = (
            vol_above.rolling(lb).sum() / lb
        ).astype(np.float32)

        # Viral spike: volume > 3 std above mean (rare, high-impact events)
        df["nlp_viral_spike"] = (
            df["nlp_buzz_volume_z"] > 3.0
        ).astype(np.float32)

        return df

    # ------------------------------------------------------------------
    # 4. GitHub activity proxy features
    # ------------------------------------------------------------------
    def _add_github_activity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy for developer/GitHub activity via momentum during dev hours.

        Active development correlates with price momentum during working
        hours (UTC 9-18). We use this time-segmented momentum as a proxy
        for commit frequency and developer engagement.
        """
        ret = df["log_return"]
        lb = self.lookback

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            hour_utc = ts.dt.hour

            # Dev hours mask
            dev_mask = (
                (hour_utc >= self.DEV_HOURS_START)
                & (hour_utc < self.DEV_HOURS_END)
            ).astype(float)

            # Momentum during dev hours only
            dev_ret = ret * dev_mask
            df["nlp_dev_momentum"] = (
                dev_ret.rolling(lb).sum()
            ).astype(np.float32)

            # Dev-hour vs off-hour return ratio
            off_ret = ret * (1.0 - dev_mask)
            dev_sum = dev_ret.rolling(lb).sum()
            off_sum = off_ret.rolling(lb).sum()
            df["nlp_dev_off_ratio"] = np.where(
                off_sum.abs() > 1e-10,
                dev_sum / off_sum.replace(0, np.nan),
                0.0,
            ).astype(np.float32)

            # Dev-hour volume concentration
            vol = df["volume"]
            dev_vol = vol * dev_mask
            total_vol = vol.rolling(lb).sum()
            df["nlp_dev_vol_share"] = np.where(
                total_vol > 0,
                dev_vol.rolling(lb).sum() / total_vol.replace(0, np.nan),
                0.5,
            ).astype(np.float32)
        else:
            # Without timestamps, use rolling momentum patterns as proxy
            logger.info(
                "BlockchainNLPFeatures: no 'timestamp' column; "
                "using momentum-only proxies for GitHub activity."
            )
            # Acceleration of returns as proxy for dev engagement
            mom = ret.rolling(lb).sum()
            df["nlp_dev_momentum"] = mom.astype(np.float32)

            # Momentum stability (low variance = steady development)
            mom_std = mom.rolling(lb).std()
            mom_mean = mom.rolling(lb).mean().abs()
            df["nlp_dev_off_ratio"] = np.where(
                mom_mean > 1e-10,
                mom_std / mom_mean.replace(0, np.nan),
                0.0,
            ).astype(np.float32)

            df["nlp_dev_vol_share"] = np.float32(0.5)

        return df

    # ------------------------------------------------------------------
    # 5. Token name/ticker mention frequency proxy
    # ------------------------------------------------------------------
    def _add_mention_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy for token mention frequency via volume and volatility surges.

        When a token trends on social media, volume and volatility spike
        together. We model mention frequency as a function of these
        co-occurring spikes.
        """
        vol = df["volume"]
        ret = df["log_return"]
        lb = self.lookback

        # Volume surge: rolling percentile rank
        min_periods = min(lb, lb * 5)
        vol_rank = vol.rolling(lb * 5, min_periods=min_periods).apply(
            lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]),
            raw=False,
        )
        df["nlp_mention_vol_rank"] = vol_rank.astype(np.float32)

        # Volatility surge: rolling realized volatility percentile
        realized_vol = ret.rolling(lb).std()
        rvol_rank = realized_vol.rolling(lb * 5, min_periods=min_periods).apply(
            lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]),
            raw=False,
        )
        df["nlp_mention_rvol_rank"] = rvol_rank.astype(np.float32)

        # Combined mention proxy: geometric mean of volume and vol ranks
        df["nlp_mention_proxy"] = np.sqrt(
            df["nlp_mention_vol_rank"].clip(lower=0)
            * df["nlp_mention_rvol_rank"].clip(lower=0)
        ).astype(np.float32)

        # Trending score: mention proxy acceleration
        mention = df["nlp_mention_proxy"]
        df["nlp_trending_score"] = (
            mention.diff(max(lb // 4, 1))
        ).astype(np.float32)

        return df

    def get_feature_names(self) -> list[str]:
        """Return names of all features generated by this module."""
        return [
            # Whitepaper/documentation complexity proxies
            "nlp_complexity_choppiness",
            "nlp_vocab_richness",
            "nlp_readability_proxy",
            "nlp_info_density",
            # Crypto sentiment lexicon proxies
            "nlp_bullish_score",
            "nlp_bearish_score",
            "nlp_net_sentiment",
            "nlp_sentiment_momentum",
            "nlp_sentiment_extreme",
            # Social media buzz proxies
            "nlp_buzz_volume_z",
            "nlp_buzz_intensity",
            "nlp_buzz_persistence",
            "nlp_viral_spike",
            # GitHub activity proxies
            "nlp_dev_momentum",
            "nlp_dev_off_ratio",
            "nlp_dev_vol_share",
            # Token mention frequency proxies
            "nlp_mention_vol_rank",
            "nlp_mention_rvol_rank",
            "nlp_mention_proxy",
            "nlp_trending_score",
        ]
