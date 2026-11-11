from app import geocode, staticmap
from app.config import settings


def test_static_map_offline_still_renders():
    png, tiles = staticmap.render([(12.87, 74.84, "large"), (12.871, 74.842, "small")],
                                  [[(12.869, 74.839), (12.872, 74.843)]])
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and tiles is False


def test_geocode_is_cached_and_offline_safe():
    assert settings.offline
    assert geocode.reverse(12.87, 74.84) == (None, None)
    from app.db import GeocodeCache, get_session
    with get_session() as s:
        s.merge(GeocodeCache(key="12.8700,74.8400", road="Light House Hill Road", area="Hampankatta"))
        s.commit()
    assert geocode.reverse(12.87001, 74.84002) == ("Light House Hill Road", "Hampankatta")
