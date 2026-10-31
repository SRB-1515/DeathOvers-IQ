"""
src/utils.py
Utilities for IPL franchise normalization, venue normalization, and chart styling.
"""

# Normalized modern franchise names mapping
FRANCHISE_MAP = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

# Major venue name consolidations (handling city suffixes and renaming)
VENUE_MAP = {
    "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium",
    "M. Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Eden Gardens, Kolkata": "Eden Gardens",
    "Wankhede Stadium, Mumbai": "Wankhede Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium, Chepauk",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium, Uppal",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association Stadium, Mohali",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association Stadium, Mohali",
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association Stadium, Mohali",
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",
    "Narendra Modi Stadium, Ahmedabad": "Narendra Modi Stadium",
    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow": "Ekana Cricket Stadium, Lucknow",
    "Dr DY Patil Sports Academy, Mumbai": "Dr DY Patil Sports Academy",
    "Maharashtra Cricket Association Stadium, Pune": "Maharashtra Cricket Association Stadium",
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium",
    "Dubai International Cricket Stadium": "Dubai International Cricket Stadium",
    "Sharjah Cricket Stadium": "Sharjah Cricket Stadium",
    "Sheikh Zayed Stadium": "Sheikh Zayed Stadium, Abu Dhabi",
    "Zayed Cricket Stadium, Abu Dhabi": "Sheikh Zayed Stadium, Abu Dhabi",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "HPCA Stadium, Dharamsala",
    "Himachal Pradesh Cricket Association Stadium": "HPCA Stadium, Dharamsala",
    "Barabati Stadium, Cuttack": "Barabati Stadium",
    "JSCA International Stadium Complex, Ranchi": "JSCA International Stadium Complex",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "ACA-VDCA Cricket Stadium, Visakhapatnam",
    "ACA-VDCA Stadium": "ACA-VDCA Cricket Stadium, Visakhapatnam",
    "Holkar Cricket Stadium, Indore": "Holkar Cricket Stadium"
}

# IPL Franchise Color Palette
TEAM_COLORS = {
    "Chennai Super Kings": "#F9CD05",
    "Mumbai Indians": "#004BA0",
    "Royal Challengers Bengaluru": "#D8232A",
    "Royal Challengers Bangalore": "#D8232A",
    "Kolkata Knight Riders": "#3A225D",
    "Delhi Capitals": "#0078BC",
    "Punjab Kings": "#ED1B24",
    "Rajasthan Royals": "#EA1A85",
    "Sunrisers Hyderabad": "#F26522",
    "Gujarat Titans": "#1B2133",
    "Lucknow Super Giants": "#A7D5E8",
    "Deccan Chargers": "#00438F",
    "Gujarat Lions": "#E04F16",
    "Rising Pune Supergiant": "#D11D9B",
    "Pune Warriors": "#2F9BE3",
    "Kochi Tuskers Kerala": "#6C1488"
}

# Theme Palette
CRICKET_ORANGE = "#F97316"
DEEP_NAVY = "#14243B"
COOL_GREY = "#F7F9FC"
BORDER_GREY = "#E7EBF0"
SLATE_TEXT = "#64748B"
SUCCESS_GREEN = "#16A36A"
WARM_AMBER = "#FBBF24"
DEFAULT_CHART_COLOR = "#F97316"

def apply_cricket_chart_theme(fig, title=None, height=380):
    """
    Applies the DeathOvers IQ premium bright chart styling to any Plotly figure.
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="'Segoe UI', -apple-system, sans-serif", color="#172337", size=12),
        margin=dict(l=40, r=30, t=55 if title else 25, b=40),
        height=height,
        hoverlabel=dict(
            bgcolor="#14243B",
            font_size=12,
            font_color="#FFFFFF",
            font_family="'Segoe UI', sans-serif"
        )
    )
    if title:
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=14, color="#14243B"),
                x=0.01,
                y=0.96
            )
        )
    fig.update_xaxes(
        gridcolor="#E7EBF0",
        zerolinecolor="#CBD5E1",
        linecolor="#E7EBF0",
        tickfont=dict(color="#64748B", size=11),
        title_font=dict(color="#14243B", size=12)
    )
    fig.update_yaxes(
        gridcolor="#E7EBF0",
        zerolinecolor="#CBD5E1",
        linecolor="#E7EBF0",
        tickfont=dict(color="#64748B", size=11),
        title_font=dict(color="#14243B", size=12)
    )
    return fig

def normalize_team(team: str, merge_rebranded: bool = True) -> str:
    """Normalize team name to standard franchise name."""
    if not team or not isinstance(team, str):
        return "Unknown Team"
    t = team.strip()
    if merge_rebranded:
        return FRANCHISE_MAP.get(t, t)
    return t

def normalize_venue(venue: str) -> str:
    """Normalize stadium / venue names removing extraneous suffixes or old names."""
    if not venue or not isinstance(venue, str):
        return "Unknown Venue"
    v = venue.strip()
    if v in VENUE_MAP:
        return VENUE_MAP[v]
    # Simple cleanup if not explicitly in map
    for raw_name, norm_name in VENUE_MAP.items():
        if raw_name.lower() in v.lower() or v.lower() in raw_name.lower():
            return norm_name
    return v
