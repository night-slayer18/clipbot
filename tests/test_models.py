"""Tests for core models."""

from __future__ import annotations

from clipbot.core.models import Asset, Clip, AssetStatus, ClipStatus, Source


def test_asset_defaults():
    a = Asset()
    assert a.source == Source.TWITCH
    assert a.status == AssetStatus.DISCOVERED
    assert a.is_ready is False


def test_clip_duration():
    c = Clip(start_time=10.0, end_time=45.0)
    assert c.duration == 35.0


def test_asset_id_unique():
    a1 = Asset()
    a2 = Asset()
    assert a1.id != a2.id
