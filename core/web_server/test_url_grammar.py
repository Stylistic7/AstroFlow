"""Lint test for [[project_url_grammar]] — every Flask rule that
references `<pipeline>` must contain the canonical view triple as a
contiguous sequence. Sub-resources of the tap (e.g. tile coords
`/<z>/<x>/<y>.png`) are allowed AFTER the view triple — global→local
ordering is preserved."""

import re


def test_every_tap_route_matches_grammar():
    from web_server.backend.app import app
    canonical_segment = "/<pipeline>/<display>/<filter>"
    pattern = re.compile(re.escape(canonical_segment) + r"(/|$)")
    offenders = [
        rule.rule for rule in app.url_map.iter_rules()
        if "<pipeline>" in rule.rule and not pattern.search(rule.rule)
    ]
    assert offenders == [], (
        f"Routes violating grammar (view triple must appear as "
        f"`{canonical_segment}` followed by `/sub-resource` or "
        f"end-of-URL): {offenders}"
    )
