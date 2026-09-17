"""Lint test for [[project_url_grammar]] — every Flask rule that
references `<pipeline>` must contain the canonical view pair
`/<pipeline>/<finish>` as a contiguous sequence; sub-resources of the view
(tile coords `/<i>/<z>/<x>/<y>.f32`) come AFTER it, global→local. No rule
addresses a tap by display name: taps are the layers finishes read, not
views (a bare `<filter>` elsewhere is a filter role, not a view)."""

import re


def test_every_view_route_matches_grammar():
    from web_server.backend.app import app
    canonical_segment = "/<pipeline>/<finish>"
    pattern = re.compile(re.escape(canonical_segment) + r"(/|$)")
    rules = [rule.rule for rule in app.url_map.iter_rules()]
    offenders = [r for r in rules if "<pipeline>" in r and not pattern.search(r)]
    assert offenders == [], (
        f"Routes violating grammar (view pair must appear as "
        f"`{canonical_segment}` followed by `/sub-resource` or end-of-URL): "
        f"{offenders}"
    )
    assert [r for r in rules if "<display>" in r] == []
