"""Lint test for [[project_url_grammar]] — every Flask rule that
references `<pipeline>` must end with the canonical view triple.
The enforcement net for tap_route."""


def test_every_tap_route_matches_grammar():
    from web_server.backend.app import app
    canonical_suffix = "/<pipeline>/<display>/<filter>"
    offenders = [
        rule.rule for rule in app.url_map.iter_rules()
        if "<pipeline>" in rule.rule and not rule.rule.endswith(canonical_suffix)
    ]
    assert offenders == [], (
        f"Routes violating grammar (must end with {canonical_suffix}): {offenders}"
    )
