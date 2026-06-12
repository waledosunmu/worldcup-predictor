"""The Odds API team name -> martj42 canonical name (only where they differ)."""

NAME_MAP = {
    "USA": "United States", "Korea Republic": "South Korea",
    "Czechia": "Czech Republic", "Türkiye": "Turkey",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Democratic Republic of the Congo": "DR Congo",
    "Republic of Ireland": "Ireland", "Côte d'Ivoire": "Ivory Coast",
}


def canon(name: str) -> str:
    return NAME_MAP.get(name, name)
