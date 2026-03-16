"""Testes para melhorias: NewsScraper, MarketContext, SignalGenerator, RealtimePipeline."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime


class TestNewsScraper:

    def test_init(self):
        from src.data.news_scraper import NewsScraper
        ns = NewsScraper()
        assert ns.max_articles == 50

    def test_keyword_sentiment(self):
        from src.data.news_scraper import NewsScraper
        ns = NewsScraper()
        # Texto positivo
        score_pos = ns._compute_sentiment("Bitcoin surges to new high in massive rally")
        assert score_pos > 0
        # Texto negativo
        score_neg = ns._compute_sentiment("Crypto crash fears as market plunges")
        assert score_neg < 0

    def test_filter_by_coin(self):
        from src.data.news_scraper import NewsScraper
        ns = NewsScraper()
        articles = [
            {"title": "Bitcoin hits new high", "text": "BTC surges past 100k"},
            {"title": "Ethereum upgrade coming", "text": "ETH 2.0 launches soon"},
            {"title": "Stock market news", "text": "S&P 500 rallies"},
        ]
        btc_articles = ns._filter_by_coin(articles, "BTC")
        assert len(btc_articles) >= 1
        assert any("Bitcoin" in a["title"] or "BTC" in a.get("text", "") for a in btc_articles)

    def test_add_news_features(self, sample_ohlcv):
        from src.data.news_scraper import NewsScraper
        from src.data.preprocessor import DataPreprocessor
        prep = DataPreprocessor()
        df = prep.clean(sample_ohlcv)
        ns = NewsScraper()
        result = ns.add_news_features(df, "BTC")
        news_cols = [c for c in result.columns if c.startswith("news_")]
        assert len(news_cols) >= 3

    def test_get_feature_names(self):
        from src.data.news_scraper import NewsScraper
        names = NewsScraper.get_feature_names()
        assert isinstance(names, list)
        assert len(names) >= 3


class TestMarketContext:

    def test_init(self):
        from src.data.market_context import MarketContextMemory
        mc = MarketContextMemory()
        assert mc.max_events == 10000

    def test_add_and_query(self):
        from src.data.market_context import MarketContextMemory
        mc = MarketContextMemory()
        mc.add_event(
            "Bitcoin crashed 15% after exchange hack",
            {"coin": "BTC", "price_change_pct": -15.0, "direction": "down"}
        )
        mc.add_event(
            "Ethereum surged 20% on ETF approval",
            {"coin": "ETH", "price_change_pct": 20.0, "direction": "up"}
        )
        results = mc.query_similar("Bitcoin drops after security breach", top_k=1)
        assert len(results) >= 1

    def test_get_feature_names(self):
        from src.data.market_context import MarketContextMemory
        names = MarketContextMemory.get_feature_names()
        assert isinstance(names, list)
        assert len(names) >= 3


class TestSignalGenerator:

    def test_generate_signal(self):
        from src.trading.signal_generator import SignalGenerator
        sg = SignalGenerator()
        predictions = {
            "xgboost_reg": {"pred": 0.02, "conf": 0.8},
            "lstm": {"pred": 0.015, "conf": 0.7},
            "lightgbm": {"pred": 0.01, "conf": 0.6},
        }
        signal = sg.generate_signal(predictions, current_price=50000.0, atr=1000.0, coin="BTC")
        assert signal.coin == "BTC"
        assert signal.direction in ("LONG", "SHORT", "HOLD")
        assert 0 <= signal.confidence <= 1
        assert 0 <= signal.position_size <= 1

    def test_hold_on_low_agreement(self):
        from src.trading.signal_generator import SignalGenerator
        sg = SignalGenerator(min_model_agreement=0.8)
        # Modelos discordam
        predictions = {
            "xgboost_reg": {"pred": 0.02, "conf": 0.7},
            "lstm": {"pred": -0.03, "conf": 0.6},
            "lightgbm": {"pred": 0.005, "conf": 0.5},
        }
        signal = sg.generate_signal(predictions, current_price=50000.0, atr=1000.0, coin="BTC")
        assert signal.direction == "HOLD"

    def test_format_signal_report(self):
        from src.trading.signal_generator import SignalGenerator
        sg = SignalGenerator()
        predictions = {
            "xgboost_reg": {"pred": 0.02, "conf": 0.8},
            "lstm": {"pred": 0.015, "conf": 0.7},
        }
        signal = sg.generate_signal(predictions, current_price=50000.0, atr=1000.0, coin="BTC")
        report = sg.format_signal_report([signal])
        assert isinstance(report, str)
        assert "BTC" in report


class TestRealtimePipeline:

    def test_init(self):
        from src.data.realtime_pipeline import RealtimePipeline
        pipeline = RealtimePipeline()
        assert pipeline.update_interval_minutes == 60

    def test_health_check(self):
        from src.data.realtime_pipeline import RealtimePipeline
        pipeline = RealtimePipeline()
        health = pipeline.health_check()
        assert isinstance(health, dict)
        assert "collector" in health or "preprocessor" in health
