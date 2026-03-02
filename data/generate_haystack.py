#!/usr/bin/env python3
"""
Needle-in-a-Haystack Generator: NIGHTMARE MODE
Generates a 200K+ token haystack that forces DSL decomposition
and requires 7+ hop reasoning with branching traps at each step.

Scenario: Annual report from the International Agricultural Research
Consortium (IARC) covering 48 field stations across 6 regions.

Usage:
    python generate_haystack.py [--target-tokens 200000] [--output haystack.txt]
"""

from __future__ import annotations

import argparse
import hashlib
import random
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Domain data for realistic filler generation
# ---------------------------------------------------------------------------

REGIONS = [
    "East African", "West African", "Central African",
    "South Asian", "Southeast Asian", "Latin American",
]

COUNTRIES: dict[str, list[str]] = {
    "East African": ["Kenya", "Tanzania", "Uganda", "Ethiopia", "Rwanda", "Mozambique"],
    "West African": ["Ghana", "Nigeria", "Senegal", "Mali", "Burkina Faso", "Côte d'Ivoire"],
    "Central African": ["Cameroon", "Democratic Republic of Congo", "Republic of Congo", "Gabon", "Central African Republic", "Chad"],
    "South Asian": ["India", "Bangladesh", "Nepal", "Sri Lanka", "Pakistan", "Myanmar"],
    "Southeast Asian": ["Vietnam", "Thailand", "Indonesia", "Philippines", "Cambodia", "Laos"],
    "Latin American": ["Brazil", "Colombia", "Peru", "Mexico", "Guatemala", "Ecuador"],
}

PREFIXES: dict[str, str] = {
    "East African": "EA", "West African": "WA", "Central African": "CA",
    "South Asian": "SA", "Southeast Asian": "SEA", "Latin American": "LA",
}

CROPS = [
    "maize", "rice", "cassava", "sorghum", "millet", "sweet potato",
    "groundnut", "cowpea", "chickpea", "lentil", "banana", "plantain",
    "teff", "yam", "pigeon pea", "finger millet",
]

SOIL_TYPES = [
    "ferralsol", "vertisol", "acrisol", "luvisol", "cambisol",
    "nitisol", "andosol", "lixisol", "arenosol", "gleysol",
]

PESTS = [
    "stem borer", "fall armyworm", "aphid", "whitefly", "leaf miner",
    "pod borer", "grain weevil", "thrips", "mealybug", "nematode",
    "bacterial blight", "rust fungus", "smut", "downy mildew", "mosaic virus",
]

EQUIPMENT = [
    "weather station", "soil moisture sensor array", "drone imaging platform",
    "automated irrigation controller", "seed processing unit", "cold storage facility",
    "greenhouse complex", "tissue culture laboratory", "grain drying system",
    "spectrophotometer", "PCR thermal cycler", "mini-combine harvester",
    "GPS-guided planter", "soil auger set", "leaf area meter",
]

PARTNER_ORGS = [
    "CGIAR", "FAO", "USAID", "DFID", "GIZ", "JICA", "KOICA",
    "Bill & Melinda Gates Foundation", "Rockefeller Foundation",
    "World Bank", "IFAD", "African Development Bank",
    "ACIAR", "IDRC", "Syngenta Foundation",
]

UNIVERSITIES = [
    "University of Nairobi", "Makerere University", "University of Ghana",
    "Indian Agricultural Research Institute", "Kasetsart University",
    "EMBRAPA", "Wageningen University", "Cornell University",
    "University of California Davis", "University of Queensland",
    "ETH Zurich", "Kyoto University", "University of São Paulo",
    "Punjab Agricultural University", "Sokoine University",
]

FIRST_NAMES = [
    "James", "Amina", "Chen", "Priya", "Carlos", "Fatima", "Kwame",
    "Linh", "Oluwaseun", "Rajesh", "Maria", "Ahmed", "Yuki", "Grace",
    "Diego", "Sunita", "Emmanuel", "Thanh", "Beatrice", "Arjun",
    "Claudia", "Ibrahim", "Keiko", "Patience", "Ravi", "Luz",
    "Kofi", "Mei", "Josephine", "Vikram", "Ana", "Moussa",
]

LAST_NAMES = [
    "Okonkwo", "Sharma", "Nguyen", "Santos", "Mensah", "Patel",
    "Diallo", "Kim", "Fernandez", "Mwangi", "Das", "Phung",
    "Andrade", "Osei", "Gupta", "Tran", "Morales", "Kamau",
    "Chakraborty", "Le", "Reyes", "Abubakar", "Rao", "Hernandez",
    "Owusu", "Singh", "Nakamura", "Pereira", "Banda", "Suresh",
]

# Unrelated domain paragraphs for variety
FILLER_DOMAINS = [
    "oceanography", "astrophysics", "medieval_history", "linguistics",
    "materials_science", "neuroscience", "urban_planning", "genetics",
    "volcanology", "cryptography", "archaeology", "epidemiology",
]


# ---------------------------------------------------------------------------
# Station and Cultivar data structures
# ---------------------------------------------------------------------------

@dataclass
class Station:
    code: str
    region: str
    country: str
    crop: str
    cultivar_code: str
    hectares: int
    staff: int
    budget_usd: int
    soil_type: str
    elevation_m: int
    annual_rainfall_mm: int
    avg_temp_c: float
    pest_resistance_score: float
    drought_tolerance: float
    yield_tons_per_ha: float
    commissioned_year: int
    partnership: Optional[str]
    partnership_hectare_pct: int  # percent of hectares in partnership trial
    equipment: list[str] = field(default_factory=list)

    @property
    def partnership_exempt_hectares(self) -> int:
        return int(self.hectares * self.partnership_hectare_pct / 100)

    @property
    def funded_hectares(self) -> int:
        return self.hectares - self.partnership_exempt_hectares


def generate_stations(rng: random.Random) -> list[Station]:
    """Generate 48 field stations (8 per region) with deterministic randomness."""
    stations: list[Station] = []
    for region in REGIONS:
        prefix = PREFIXES[region]
        countries = COUNTRIES[region]
        for i in range(1, 9):
            code = f"{prefix}-{i:02d}"
            country = countries[i % len(countries)]
            crop = rng.choice(CROPS)
            # Generate a realistic cultivar code
            cultivar_prefix = crop[:2].upper()
            cultivar_num = rng.randint(100, 999)
            cultivar_code = f"{cultivar_prefix}-{cultivar_num}"

            hectares = rng.randint(80, 600)
            staff = rng.randint(8, 45)
            budget = rng.randint(400_000, 3_500_000)
            soil = rng.choice(SOIL_TYPES)
            elevation = rng.randint(200, 2400)
            rainfall = rng.randint(400, 2200)
            temp = round(rng.uniform(18.0, 32.0), 1)
            pest_score = round(rng.uniform(3.0, 9.5), 1)
            drought = round(rng.uniform(4.0, 9.0), 1)
            yield_t = round(rng.uniform(1.5, 8.0), 1)
            year = rng.randint(1965, 2015)
            partner = rng.choice(PARTNER_ORGS + [None, None, None])
            partner_pct = rng.choice([0, 0, 0, 10, 15, 20]) if partner else 0
            equip = rng.sample(EQUIPMENT, k=rng.randint(3, 7))

            stations.append(Station(
                code=code, region=region, country=country, crop=crop,
                cultivar_code=cultivar_code, hectares=hectares, staff=staff,
                budget_usd=budget, soil_type=soil, elevation_m=elevation,
                annual_rainfall_mm=rainfall, avg_temp_c=temp,
                pest_resistance_score=pest_score, drought_tolerance=drought,
                yield_tons_per_ha=yield_t, commissioned_year=year,
                partnership=partner, partnership_hectare_pct=partner_pct,
                equipment=equip,
            ))
    return stations


# ---------------------------------------------------------------------------
# Needle configuration — THE ACTUAL PUZZLE
# ---------------------------------------------------------------------------

# We override specific stations to create the puzzle chain.
# These values are set AFTER random generation to ensure the chain works.

def configure_needles(stations: list[Station]) -> dict:
    """
    Overwrite specific station attributes to create the 8-hop puzzle.

    PUZZLE CHAIN:
    1. EA-03 has the highest pest resistance in "East African" cluster (8.9)
    2. BUT EA-03 is reclassified to "Central African" in a governance paragraph
    3. After reclassification, EA-07 has the highest East African score (8.4)
    4. EA-07's primary cultivar is MA-447
    5. MA-447's drought tolerance was ORIGINALLY 7.3
    6. But the 2023 re-evaluation revised MA-447 to 6.6
    7. Cultivars with drought tolerance < 7.0 get "Tier B" funding: $38,500/hectare
       (Tier A is >= 7.0: $54,200/hectare — the decoy)
    8. EA-07 has 420 hectares, but 15% (63 ha) are in a JICA partnership trial (exempt)
    9. ANSWER: (420 - 63) * $38,500 = 357 * $38,500 = $13,744,500

    DECOY PATHS:
    - Use EA-03 (wrong station after reclassification): different cultivar, different answer
    - Use original drought tolerance 7.3 (>= 7.0 → Tier A): 357 * $54,200 = $19,349,400
    - Forget partnership exemption: 420 * $38,500 = $16,170,000
    - Use EA-03's cultivar: completely wrong drought tolerance
    - Use Tier A rate with full hectares: 420 * $54,200 = $22,764,000
    """

    # Find and configure key stations
    needle_stations: dict[str, Station] = {}
    for s in stations:
        if s.code == "EA-03":
            s.pest_resistance_score = 8.9  # highest in East African — but will be reclassified
            s.crop = "cassava"
            s.cultivar_code = "CA-612"
            s.drought_tolerance = 7.8
            s.hectares = 290
            s.partnership = "USAID"
            s.partnership_hectare_pct = 20
            needle_stations["EA-03"] = s
        elif s.code == "EA-07":
            s.pest_resistance_score = 8.4  # second highest, becomes first after reclassification
            s.crop = "maize"
            s.cultivar_code = "MA-447"
            s.drought_tolerance = 7.3  # original — will be revised to 6.6
            s.hectares = 420
            s.partnership = "JICA"
            s.partnership_hectare_pct = 15
            s.country = "Tanzania"
            needle_stations["EA-07"] = s
        elif s.code == "EA-05":
            s.pest_resistance_score = 8.1  # third highest — another decoy
            s.crop = "maize"
            s.cultivar_code = "MA-331"
            s.drought_tolerance = 5.9
            needle_stations["EA-05"] = s

    # Ensure other East African stations have lower pest resistance
    for s in stations:
        if s.region == "East African" and s.code not in ("EA-03", "EA-07", "EA-05"):
            s.pest_resistance_score = round(min(s.pest_resistance_score, 7.8), 1)

    return {
        "target_station": "EA-07",
        "decoy_station": "EA-03",
        "cultivar": "MA-447",
        "original_drought": 7.3,
        "revised_drought": 6.6,
        "tier_a_rate": 54_200,
        "tier_b_rate": 38_500,
        "total_hectares": 420,
        "exempt_pct": 15,
        "exempt_hectares": 63,
        "funded_hectares": 357,
        "correct_answer": 357 * 38_500,  # $13,744,500
        "decoy_answers": {
            "wrong_station_ea03": "different cultivar → different calculation entirely",
            "original_drought_tier_a": 357 * 54_200,       # $19,349,400
            "no_exemption_tier_b": 420 * 38_500,            # $16,170,000
            "no_exemption_tier_a": 420 * 54_200,            # $22,764,000
            "wrong_station_full_calc": 290 * 0.80 * 54_200, # $12,574,400 (EA-03 path)
        },
    }


# ---------------------------------------------------------------------------
# Paragraph generators
# ---------------------------------------------------------------------------

def gen_station_description(s: Station, rng: random.Random) -> str:
    """Generate a station description paragraph."""
    templates = [
        (
            f"Station {s.code}, located in {s.country} at an elevation of {s.elevation_m} meters "
            f"above sea level, focuses primarily on {s.crop} research across {s.hectares} hectares "
            f"of experimental plots. The facility was commissioned in {s.commissioned_year} and "
            f"currently employs {s.staff} full-time research and support staff. The station's "
            f"annual operating budget for the current reporting period is ${s.budget_usd:,}. "
            f"The dominant soil type across the experimental plots is {s.soil_type}, with an "
            f"average annual rainfall of {s.annual_rainfall_mm} millimeters and mean annual "
            f"temperature of {s.avg_temp_c} degrees Celsius. The station's primary cultivar "
            f"under evaluation is designated {s.cultivar_code}."
        ),
        (
            f"The {s.code} research station in {s.country} operates on {s.hectares} hectares of "
            f"land at {s.elevation_m} meters elevation, where researchers have been studying "
            f"{s.crop} varieties since the facility's establishment in {s.commissioned_year}. "
            f"With a team of {s.staff} personnel and a budget of ${s.budget_usd:,}, the station "
            f"maintains active breeding programs focused on cultivar {s.cultivar_code}. "
            f"Environmental conditions at the site include {s.annual_rainfall_mm} millimeters "
            f"of annual precipitation, a mean temperature of {s.avg_temp_c} degrees Celsius, "
            f"and predominantly {s.soil_type} soils."
        ),
        (
            f"Situated in the {s.country} highlands at {s.elevation_m} meters, station {s.code} "
            f"manages {s.hectares} hectares dedicated to {s.crop} improvement. The facility, "
            f"operational since {s.commissioned_year}, supports {s.staff} staff members on an "
            f"annual budget of ${s.budget_usd:,}. Key cultivar {s.cultivar_code} has been the "
            f"centerpiece of the station's breeding program. The site receives approximately "
            f"{s.annual_rainfall_mm} millimeters of rainfall annually, with {s.soil_type} soils "
            f"and average temperatures of {s.avg_temp_c} degrees Celsius."
        ),
    ]
    return rng.choice(templates)


def gen_station_performance(s: Station, rng: random.Random) -> str:
    """Generate a performance/yield paragraph for a station."""
    pest_context = rng.choice([
        f"pest resistance evaluations gave cultivar {s.cultivar_code} a composite score of {s.pest_resistance_score}",
        f"the integrated pest management trials at {s.code} resulted in a resistance score of {s.pest_resistance_score} for the primary cultivar",
        f"resistance screening at {s.code} rated {s.cultivar_code} at {s.pest_resistance_score} on the standardized 10-point scale",
    ])
    drought_context = rng.choice([
        f"Drought tolerance for {s.cultivar_code} was assessed at {s.drought_tolerance} under controlled deficit irrigation",
        f"Water stress trials at {s.code} yielded a drought tolerance rating of {s.drought_tolerance} for cultivar {s.cultivar_code}",
        f"The cultivar's drought tolerance index stood at {s.drought_tolerance} based on multi-season evaluation",
    ])
    return (
        f"During the 2022 growing season, {s.code} achieved an average yield of "
        f"{s.yield_tons_per_ha} metric tons per hectare for {s.crop}, which "
        f"{'exceeded' if s.yield_tons_per_ha > 4.0 else 'fell below'} the regional "
        f"target of 4.0 metric tons per hectare. Field-level {pest_context}. "
        f"{drought_context}. The station's overall performance ranking within the "
        f"{s.region} cluster placed it "
        f"{'in the top quartile' if s.yield_tons_per_ha > 5.5 else 'in the middle tier' if s.yield_tons_per_ha > 3.5 else 'in the lower tier'} "
        f"for the reporting period."
    )


def gen_station_equipment(s: Station, rng: random.Random) -> str:
    """Generate an equipment/infrastructure paragraph."""
    equip_str = ", ".join(s.equipment[:-1]) + f", and {s.equipment[-1]}" if len(s.equipment) > 1 else s.equipment[0]
    maint_cost = rng.randint(15_000, 120_000)
    return (
        f"The infrastructure inventory at station {s.code} includes a {equip_str}. "
        f"Total equipment maintenance costs for the reporting period were ${maint_cost:,}. "
        f"A capital improvement plan submitted in Q2 2022 requested funding for "
        f"{'a new ' + rng.choice(EQUIPMENT) if rng.random() > 0.5 else 'upgrades to the existing ' + rng.choice(s.equipment)}, "
        f"estimated at ${rng.randint(50_000, 300_000):,}. The station's laboratory "
        f"facilities were last audited in {rng.randint(2018, 2022)} and received "
        f"{'full accreditation' if rng.random() > 0.3 else 'conditional accreditation with minor findings'}."
    )


def gen_station_partnership(s: Station, rng: random.Random) -> str:
    """Generate a partnership paragraph (only for stations with partners)."""
    if not s.partnership:
        return ""
    return (
        f"Station {s.code} maintains an active research partnership with {s.partnership} "
        f"under a memorandum of understanding renewed in {rng.randint(2019, 2023)}. "
        f"The partnership allocates {s.partnership_hectare_pct} percent of the station's "
        f"cultivable area — approximately {s.partnership_exempt_hectares} hectares — to "
        f"collaborative trial plots. These plots are funded separately through the partner's "
        f"budget and are exempt from the Consortium's standard per-hectare funding formula. "
        f"The partnership's focus is on {rng.choice(['improved germplasm evaluation', 'integrated soil fertility management', 'climate adaptation trials', 'post-harvest loss reduction', 'farmer participatory variety selection'])}. "
        f"Joint publications from the collaboration totaled {rng.randint(2, 12)} peer-reviewed "
        f"papers during the reporting period."
    )


def gen_personnel_paragraph(s: Station, rng: random.Random) -> str:
    """Generate a personnel update paragraph."""
    name1 = f"Dr. {rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    name2 = f"Dr. {rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    uni = rng.choice(UNIVERSITIES)
    return (
        f"Personnel changes at {s.code} during the reporting period included the appointment "
        f"of {name1} as station director, replacing {name2} who transferred to the "
        f"Consortium's regional coordination office. {name1} holds a doctorate in "
        f"{rng.choice(['plant breeding', 'agronomy', 'crop physiology', 'soil science', 'agricultural economics'])} "
        f"from {uni} and has {rng.randint(8, 25)} years of experience in {s.crop} research. "
        f"The station also recruited {rng.randint(1, 4)} new research associates and "
        f"{rng.randint(0, 2)} visiting scientists during the period."
    )


def gen_weather_paragraph(s: Station, rng: random.Random) -> str:
    """Generate a weather/climate paragraph."""
    anomaly = round(rng.uniform(-2.5, 2.5), 1)
    rain_pct = rng.randint(-30, 40)
    return (
        f"Meteorological conditions at {s.code} during the 2022 growing season deviated "
        f"from the 30-year baseline, with a temperature anomaly of {'+' if anomaly > 0 else ''}{anomaly} "
        f"degrees Celsius and precipitation {'exceeding' if rain_pct > 0 else 'falling below'} "
        f"the long-term average by {abs(rain_pct)} percent ({s.annual_rainfall_mm + int(s.annual_rainfall_mm * rain_pct / 100)} "
        f"millimeters recorded versus {s.annual_rainfall_mm} millimeters expected). "
        f"{'Extended dry spells in the vegetative phase reduced stand establishment at several trial plots.' if rain_pct < -15 else ''} "
        f"{'Above-normal rainfall during the grain-filling stage raised concerns about fungal disease pressure.' if rain_pct > 20 else ''} "
        f"Soil moisture measurements at the 30-centimeter depth averaged "
        f"{round(rng.uniform(12.0, 38.0), 1)} percent volumetric water content during the critical reproductive stage."
    )


def gen_unrelated_filler(rng: random.Random, idx: int) -> str:
    """Generate a filler paragraph from an unrelated domain to add noise."""
    domain = FILLER_DOMAINS[idx % len(FILLER_DOMAINS)]
    # These are templates with randomized numbers to create unique, plausible paragraphs
    fillers = {
        "oceanography": (
            f"Bathymetric surveys of the {rng.choice(['Mariana', 'Sunda', 'Philippine', 'Tonga', 'Kermadec'])} "
            f"Trench conducted during the {rng.randint(2018, 2023)} expedition recorded maximum depths of "
            f"{rng.randint(8200, 10994)} meters using multibeam sonar systems with a vertical accuracy "
            f"of approximately {rng.randint(5, 25)} meters. Water column profiles revealed a thermocline "
            f"at {rng.randint(100, 400)} meters depth, below which temperatures decreased to approximately "
            f"{round(rng.uniform(1.0, 4.0), 1)} degrees Celsius at the abyssal floor. Dissolved oxygen "
            f"concentrations showed a minimum of {round(rng.uniform(0.5, 2.5), 1)} milligrams per liter "
            f"at approximately {rng.randint(600, 1200)} meters — the oxygen minimum zone — before "
            f"increasing slightly at greater depths due to deep water circulation patterns."
        ),
        "astrophysics": (
            f"Spectroscopic analysis of the exoplanet {rng.choice(['Kepler', 'TOI', 'WASP', 'HAT-P'])}-"
            f"{rng.randint(10, 999)}{rng.choice(['b', 'c', 'd'])} using the James Webb Space Telescope's "
            f"NIRSpec instrument detected absorption features consistent with "
            f"{rng.choice(['water vapor', 'carbon dioxide', 'methane', 'sulfur dioxide'])} in the planet's "
            f"atmosphere at a confidence level of {round(rng.uniform(2.5, 5.5), 1)} sigma. The planet "
            f"orbits its host star at a distance of {round(rng.uniform(0.02, 1.5), 2)} astronomical units "
            f"with an orbital period of {round(rng.uniform(1.5, 365.0), 1)} days. Its estimated surface "
            f"temperature of {rng.randint(200, 2500)} kelvin places it "
            f"{'within' if rng.random() > 0.5 else 'outside'} the habitable zone of the system."
        ),
        "medieval_history": (
            f"Charter records from the {rng.choice(['Benedictine', 'Cistercian', 'Augustinian', 'Franciscan'])} "
            f"monastery at {rng.choice(['Cluny', 'Fontenay', 'Rievaulx', 'Melk', 'Mont Saint-Michel'])} "
            f"document agricultural output for the year {rng.randint(1100, 1400)}. The monastery's demesne "
            f"lands encompassed approximately {rng.randint(200, 1500)} hectares, producing an estimated "
            f"{rng.randint(100, 800)} tonnes of grain — primarily {rng.choice(['wheat', 'barley', 'rye', 'oats'])} — "
            f"and supporting a population of {rng.randint(40, 300)} monks and lay brothers. Tithe records "
            f"suggest an average yield of {round(rng.uniform(0.5, 2.0), 1)} tonnes per hectare, "
            f"consistent with other monastic estates of the period."
        ),
        "linguistics": (
            f"Computational analysis of a {rng.randint(50, 500)}-million-word corpus of "
            f"{rng.choice(['Mandarin', 'Hindi', 'Swahili', 'Turkish', 'Tagalog', 'Quechua'])} text "
            f"identified approximately {rng.randint(30000, 120000)} distinct lemmas, with the most "
            f"frequent {rng.randint(1000, 5000)} accounting for roughly {rng.randint(80, 95)} percent "
            f"of all tokens. Zipf's law held with an exponent of {round(rng.uniform(0.9, 1.3), 2)} across "
            f"the frequency distribution. Morphological complexity, measured by the ratio of word forms to "
            f"lemmas, was approximately {round(rng.uniform(1.2, 8.5), 1)}, reflecting the language's "
            f"{'agglutinative' if rng.random() > 0.5 else 'fusional'} typology."
        ),
        "materials_science": (
            f"Tensile testing of a novel {rng.choice(['titanium-aluminum', 'nickel-chromium', 'iron-manganese', 'cobalt-tungsten'])} "
            f"alloy (composition: {rng.randint(60, 85)} percent base metal, {rng.randint(5, 20)} percent "
            f"secondary element, balance trace additions) yielded an ultimate tensile strength of "
            f"{rng.randint(400, 1800)} megapascals and an elongation at fracture of {rng.randint(2, 35)} "
            f"percent at room temperature. Fatigue life testing at {rng.randint(40, 80)} percent of yield "
            f"strength showed failure at {rng.randint(50000, 5000000):,} cycles. Scanning electron "
            f"microscopy of the fracture surface revealed {'transgranular cleavage' if rng.random() > 0.5 else 'intergranular failure'}, "
            f"suggesting {'brittle' if rng.random() > 0.5 else 'ductile'} behavior at the microstructural level."
        ),
        "neuroscience": (
            f"Functional MRI data from {rng.randint(30, 200)} participants performing a "
            f"{rng.choice(['working memory', 'decision-making', 'emotional regulation', 'language processing'])} "
            f"task revealed significant activation in the {rng.choice(['dorsolateral prefrontal cortex', 'anterior cingulate', 'inferior frontal gyrus', 'temporal-parietal junction'])} "
            f"(p < {rng.choice(['0.001', '0.005', '0.01'])}, FWE-corrected). The BOLD signal change was "
            f"{round(rng.uniform(0.3, 2.5), 1)} percent above baseline, with a peak at approximately "
            f"{round(rng.uniform(4.0, 8.0), 1)} seconds post-stimulus. Connectivity analysis showed "
            f"increased coupling between this region and the {'hippocampus' if rng.random() > 0.5 else 'amygdala'} "
            f"during high-demand trials (z = {round(rng.uniform(2.0, 5.0), 1)})."
        ),
        "urban_planning": (
            f"The {rng.choice(['transit-oriented development', 'mixed-use zoning', 'green infrastructure', 'complete streets'])} "
            f"initiative in {rng.choice(['Medellín', 'Copenhagen', 'Curitiba', 'Singapore', 'Vienna', 'Barcelona'])} "
            f"covered approximately {rng.randint(5, 50)} square kilometers and affected an estimated "
            f"{rng.randint(50000, 500000):,} residents. The project budget of ${rng.randint(20, 800)} million "
            f"included {rng.randint(5, 40)} percent allocated to public green space. Post-implementation surveys "
            f"showed a {rng.randint(5, 35)} percent increase in public transit ridership and a "
            f"{rng.randint(8, 25)} percent reduction in private vehicle trips within the project area. "
            f"Property values within {rng.randint(200, 800)} meters of the intervention increased by an "
            f"average of {rng.randint(5, 30)} percent over the {rng.randint(3, 8)}-year monitoring period."
        ),
        "genetics": (
            f"Genome-wide association analysis of {rng.randint(5000, 100000):,} individuals from the "
            f"{rng.choice(['UK Biobank', 'All of Us', 'BioBank Japan', 'China Kadoorie Biobank'])} cohort "
            f"identified {rng.randint(15, 200)} loci associated with {rng.choice(['height', 'BMI', 'blood pressure', 'type 2 diabetes risk', 'LDL cholesterol'])} "
            f"at genome-wide significance (p < 5 × 10⁻⁸). The lead SNP, rs{rng.randint(1000000, 99999999)}, "
            f"located in an intron of the {rng.choice(['FTO', 'MC4R', 'PCSK9', 'APOE', 'TCF7L2'])} gene, "
            f"explained approximately {round(rng.uniform(0.1, 3.0), 1)} percent of phenotypic variance. "
            f"Polygenic risk scores constructed from the significant loci achieved an area under the ROC "
            f"curve of {round(rng.uniform(0.55, 0.80), 2)} in an independent validation cohort."
        ),
        "volcanology": (
            f"Continuous GPS monitoring of {rng.choice(['Mount Etna', 'Kīlauea', 'Popocatépetl', 'Mount Merapi', 'Sakurajima'])} "
            f"detected ground deformation of approximately {rng.randint(2, 50)} millimeters per month "
            f"at a station {rng.randint(2, 15)} kilometers from the summit during the {rng.randint(2020, 2023)} "
            f"monitoring period. The deformation pattern was consistent with magma accumulation at a depth "
            f"of approximately {rng.randint(3, 20)} kilometers. Seismic data showed an average of "
            f"{rng.randint(5, 200)} volcano-tectonic earthquakes per day, with magnitudes up to "
            f"M{round(rng.uniform(1.5, 4.5), 1)}. SO₂ flux measurements from DOAS traverses averaged "
            f"{rng.randint(100, 5000)} tonnes per day, {'above' if rng.random() > 0.5 else 'near'} "
            f"the baseline for the volcano."
        ),
        "cryptography": (
            f"Benchmarking of the {rng.choice(['CRYSTALS-Kyber', 'CRYSTALS-Dilithium', 'SPHINCS+', 'BIKE', 'Classic McEliece'])} "
            f"post-quantum cryptographic algorithm on an ARM Cortex-{rng.choice(['A72', 'A76', 'M4', 'M7'])} processor "
            f"showed key generation times of {rng.randint(50, 5000)} microseconds, encapsulation "
            f"(or signing) at {rng.randint(80, 8000)} microseconds, and decapsulation (or verification) "
            f"at {rng.randint(60, 6000)} microseconds. Public key sizes were {rng.randint(800, 1500000)} "
            f"bytes, with ciphertext (or signature) sizes of {rng.randint(700, 50000)} bytes. Compared to "
            f"RSA-{rng.choice(['2048', '3072', '4096'])}, the post-quantum scheme required approximately "
            f"{rng.randint(2, 50)} times more bandwidth but {'comparable' if rng.random() > 0.5 else str(rng.randint(2, 10)) + ' times more'} "
            f"computation."
        ),
        "archaeology": (
            f"Excavations at the {rng.choice(['Neolithic', 'Bronze Age', 'Iron Age', 'Hellenistic'])} site "
            f"of {rng.choice(['Tell', 'Çatal', 'Mohenjo', 'Chan Chan', 'Great Zimbabwe'])} "
            f"during the {rng.randint(2019, 2023)} season uncovered {rng.randint(3, 50)} stratigraphic "
            f"layers spanning approximately {rng.randint(200, 3000)} years. Radiocarbon dating of charcoal "
            f"samples from the lowest occupation layer yielded a calibrated date of {rng.randint(3000, 9000)} "
            f"± {rng.randint(30, 200)} BCE. Ceramic typology suggested trade connections extending over "
            f"{rng.randint(50, 500)} kilometers. Faunal remains included {rng.randint(2000, 20000):,} "
            f"identifiable bone fragments, with {rng.choice(['cattle', 'sheep/goat', 'pig', 'deer'])} "
            f"comprising approximately {rng.randint(30, 70)} percent of the assemblage."
        ),
        "epidemiology": (
            f"A prospective cohort study of {rng.randint(5000, 200000):,} participants followed for a median "
            f"of {round(rng.uniform(3.0, 15.0), 1)} years found that individuals in the highest quartile "
            f"of {rng.choice(['ultra-processed food consumption', 'physical activity', 'sleep duration', 'vegetable intake'])} "
            f"had a hazard ratio of {round(rng.uniform(0.5, 2.5), 2)} (95% CI: {round(rng.uniform(0.4, 1.8), 2)}-"
            f"{round(rng.uniform(1.2, 3.5), 2)}) for {rng.choice(['all-cause mortality', 'cardiovascular events', 'cancer incidence', 'type 2 diabetes'])} "
            f"compared to the lowest quartile, after adjustment for age, sex, BMI, smoking status, and "
            f"socioeconomic factors. The population-attributable fraction was estimated at "
            f"{round(rng.uniform(2.0, 20.0), 1)} percent."
        ),
    }
    return fillers.get(domain, fillers["oceanography"])


# ---------------------------------------------------------------------------
# Needle paragraph generators — these are HAND-CRAFTED for the puzzle
# ---------------------------------------------------------------------------

def needle_1_ea03_performance(s_ea03: Station) -> str:
    """Needle 1: EA-03 has highest pest resistance in East African cluster."""
    return (
        f"During the 2022 growing season, {s_ea03.code} achieved an average yield of "
        f"{s_ea03.yield_tons_per_ha} metric tons per hectare for {s_ea03.crop}. "
        f"Resistance screening at {s_ea03.code} rated cultivar {s_ea03.cultivar_code} at "
        f"{s_ea03.pest_resistance_score} on the standardized 10-point pest resistance scale "
        f"— the highest score recorded among all stations in the East African cluster "
        f"for the 2022 evaluation cycle. Water stress trials yielded a drought tolerance "
        f"index of {s_ea03.drought_tolerance} for {s_ea03.cultivar_code}. The station's overall "
        f"performance was rated as exemplary by the regional review committee."
    )


def needle_2_ea07_description(s_ea07: Station) -> str:
    """Needle 2: EA-07 description with cultivar MA-447 and original drought tolerance."""
    return (
        f"Station {s_ea07.code}, located in {s_ea07.country} at an elevation of "
        f"{s_ea07.elevation_m} meters, manages {s_ea07.hectares} hectares of {s_ea07.crop} "
        f"trials. The facility was commissioned in {s_ea07.commissioned_year} and supports "
        f"{s_ea07.staff} staff on an annual budget of ${s_ea07.budget_usd:,}. The station's "
        f"primary cultivar under evaluation is {s_ea07.cultivar_code}, which recorded a pest "
        f"resistance score of {s_ea07.pest_resistance_score} during the 2022 screening — the "
        f"second-highest in the East African cluster, behind only {s_ea07.code.replace('07', '03')}. "
        f"The original drought tolerance assessment for {s_ea07.cultivar_code}, conducted in 2021, "
        f"yielded an index of {s_ea07.drought_tolerance}."
    )


def needle_3_ea07_partnership(s_ea07: Station) -> str:
    """Needle 3: EA-07's JICA partnership with 15% exemption."""
    return (
        f"Station {s_ea07.code} maintains a collaborative research agreement with {s_ea07.partnership} "
        f"under a framework renewed in 2021. The agreement designates {s_ea07.partnership_hectare_pct} "
        f"percent of the station's cultivable area — approximately "
        f"{s_ea07.partnership_exempt_hectares} hectares of the total {s_ea07.hectares} — for "
        f"joint trial plots investigating climate-resilient cropping systems. These partnership "
        f"plots receive separate funding through {s_ea07.partnership}'s bilateral budget and are "
        f"therefore exempt from the Consortium's standard per-hectare funding calculation. "
        f"The remaining {s_ea07.funded_hectares} hectares fall under the Consortium's funding formula."
    )


def needle_4_reclassification() -> str:
    """Needle 4: EA-03 reclassified from East African to Central African cluster."""
    return (
        "Following the Consortium's governance review completed in September 2022, several "
        "stations were reassigned to different regional clusters to better reflect geographic "
        "and agroecological alignment. The most significant change affected station EA-03, "
        "which had historically been classified within the East African cluster despite its "
        "location in a transitional zone. Effective January 1, 2023, EA-03 was formally "
        "reclassified under the Central African cluster for all administrative, reporting, "
        "and funding purposes. The station retains its original EA-prefix designation to "
        "maintain continuity in the research database. This reclassification adjusts the "
        "composition of the East African cluster for all forward-looking performance "
        "comparisons, rankings, and funding allocations."
    )


def needle_5_drought_revision() -> str:
    """Needle 5: Cultivar MA-447's drought tolerance revised downward."""
    return (
        "The Consortium's 2023 cultivar re-evaluation program, which applied updated "
        "screening protocols incorporating longer stress periods and multi-environment "
        "trial data, resulted in revised drought tolerance indices for 23 cultivars "
        "across the network. Notable revisions included cultivar MA-447, whose drought "
        "tolerance index was adjusted from its original assessment of 7.3 downward to "
        "6.6 — a reduction attributed to the cultivar's poor performance under extended "
        "terminal drought conditions observed at multiple East African trial sites during "
        "the 2022 season. Cultivar SO-218 was similarly revised from 6.1 to 5.4, while "
        "cultivar RI-705 saw an upward adjustment from 5.8 to 6.9 based on strong "
        "performance in South Asian water-limited environments."
    )


def needle_6_funding_formula() -> str:
    """Needle 6: Tier A/B funding rates based on drought tolerance threshold."""
    return (
        "The Consortium's per-hectare funding formula for the 2023-2025 funding cycle "
        "employs a two-tier structure linked to the drought tolerance index of each "
        "station's primary cultivar, as determined by the most recent re-evaluation. "
        "Stations whose primary cultivar holds a drought tolerance index of 7.0 or above "
        "qualify for Tier A funding at a rate of $54,200 per hectare per annum. Stations "
        "whose primary cultivar falls below a drought tolerance index of 7.0 receive "
        "Tier B funding at a rate of $38,500 per hectare per annum. The per-hectare "
        "calculation applies only to land under the Consortium's direct funding — hectares "
        "allocated to bilateral partnership trials that receive separate external funding "
        "are excluded from the formula. The tier classification uses the most recent "
        "drought tolerance assessment, including any revisions from the 2023 re-evaluation."
    )


def needle_7_board_question_context() -> str:
    """Needle 7: Board asks about funding for best-performing East African station."""
    return (
        "During the Q1 2023 board meeting, the finance committee requested a worked "
        "example of the new funding formula as applied to the highest-performing station "
        "in the East African cluster — specifically, the station whose primary cultivar "
        "achieved the highest pest resistance score among currently classified East African "
        "stations. The committee noted that this calculation should reflect the current "
        "cluster composition (incorporating any reclassifications), the most recent drought "
        "tolerance re-evaluation results, the applicable funding tier, and the exclusion of "
        "any partnership-exempt hectares. The Secretariat was directed to prepare this "
        "calculation for the Q2 meeting."
    )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per word for English."""
    return int(len(text.split()) * 1.33)


def generate_haystack(
    target_tokens: int = 200_000,
    seed: int = 42,
    output_path: str = "haystack.txt",
) -> dict:
    """Generate the full haystack with embedded needles."""
    rng = random.Random(seed)
    stations = generate_stations(rng)
    puzzle = configure_needles(stations)

    # Find needle stations
    ea03 = next(s for s in stations if s.code == "EA-03")
    ea07 = next(s for s in stations if s.code == "EA-07")

    # Generate all needle paragraphs
    needles = {
        # position as fraction of total paragraphs
        0.08: ("NEEDLE_1_EA03_PERF", needle_1_ea03_performance(ea03)),
        0.15: ("NEEDLE_2_EA07_DESC", needle_2_ea07_description(ea07)),
        0.22: ("NEEDLE_3_EA07_PARTNER", needle_3_ea07_partnership(ea07)),
        0.45: ("NEEDLE_4_RECLASS", needle_4_reclassification()),
        0.62: ("NEEDLE_5_DROUGHT_REV", needle_5_drought_revision()),
        0.78: ("NEEDLE_6_FUNDING", needle_6_funding_formula()),
        0.88: ("NEEDLE_7_BOARD_Q", needle_7_board_question_context()),
    }

    # Generate filler paragraphs until we hit target token count
    paragraphs: list[str] = []
    filler_idx = 0

    # Opening paragraph
    paragraphs.append(
        "INTERNATIONAL AGRICULTURAL RESEARCH CONSORTIUM — ANNUAL REPORT 2022\n\n"
        "This report summarizes research activities, performance metrics, financial data, "
        "and governance decisions across the Consortium's global network of 48 field stations "
        "operating in six regional clusters: East African, West African, Central African, "
        "South Asian, Southeast Asian, and Latin American. The reporting period covers "
        "January 1, 2022 through December 31, 2022, with governance updates through Q1 2023."
    )

    while estimate_tokens("\n\n".join(paragraphs)) < target_tokens:
        # Cycle through station-based and filler paragraphs
        station = stations[filler_idx % len(stations)]
        para_type = filler_idx % 8

        if para_type == 0:
            paragraphs.append(gen_station_description(station, rng))
        elif para_type == 1:
            paragraphs.append(gen_station_performance(station, rng))
        elif para_type == 2:
            paragraphs.append(gen_station_equipment(station, rng))
        elif para_type == 3:
            p = gen_station_partnership(station, rng)
            if p:
                paragraphs.append(p)
        elif para_type == 4:
            paragraphs.append(gen_personnel_paragraph(station, rng))
        elif para_type == 5:
            paragraphs.append(gen_weather_paragraph(station, rng))
        elif para_type == 6:
            paragraphs.append(gen_unrelated_filler(rng, filler_idx))
        elif para_type == 7:
            paragraphs.append(gen_unrelated_filler(rng, filler_idx + 3))

        filler_idx += 1

    # Insert needles at target positions
    total = len(paragraphs)
    inserted = 0
    needle_positions: dict[str, int] = {}
    for frac in sorted(needles.keys()):
        label, text = needles[frac]
        pos = int(frac * total) + inserted
        paragraphs.insert(pos, text)
        needle_positions[label] = pos
        inserted += 1

    # Build final text
    final_text = "\n\n".join(paragraphs)
    actual_tokens = estimate_tokens(final_text)

    # Write haystack
    Path(output_path).write_text(final_text, encoding="utf-8")

    return {
        "output_path": output_path,
        "total_paragraphs": len(paragraphs),
        "estimated_tokens": actual_tokens,
        "word_count": len(final_text.split()),
        "char_count": len(final_text),
        "needle_positions": needle_positions,
        "puzzle": puzzle,
    }


# ---------------------------------------------------------------------------
# Question / Answer / Scoring
# ---------------------------------------------------------------------------

QUESTION = (
    "Based on the Consortium's current funding formula, what is the total annual "
    "Consortium-funded amount (in US dollars) for the East African cluster station "
    "whose primary cultivar achieved the highest pest resistance score — taking into "
    "account any cluster reclassifications, the most recent drought tolerance "
    "re-evaluation, the applicable funding tier, and the exclusion of partnership-"
    "exempt hectares?"
)

CORRECT_ANSWER = 13_744_500  # 357 * $38,500

REASONING_CHAIN = """
REQUIRED REASONING CHAIN (8 hops):

1. IDENTIFY: Which East African station has the highest pest resistance?
   → EA-03 scored 8.9, EA-07 scored 8.4, EA-05 scored 8.1
   → Naive answer: EA-03

2. CHECK RECLASSIFICATION: Is EA-03 still in the East African cluster?
   → NO. EA-03 was reclassified to Central African effective Jan 1, 2023.
   → Current highest in East African: EA-07 (8.4)

3. IDENTIFY CULTIVAR: What is EA-07's primary cultivar?
   → MA-447

4. FIND DROUGHT TOLERANCE: What is MA-447's drought tolerance?
   → Original: 7.3 (from 2021 assessment)
   → REVISED: 6.6 (from 2023 re-evaluation)
   → Must use revised value per funding formula rules

5. DETERMINE TIER: Which funding tier applies?
   → 6.6 < 7.0 → Tier B ($38,500/hectare)
   → NOT Tier A ($54,200/hectare) — a common trap using original 7.3 ≥ 7.0

6. FIND TOTAL HECTARES: How many hectares does EA-07 have?
   → 420 hectares total

7. EXCLUDE PARTNERSHIP HECTARES: How many are exempt?
   → 15% in JICA partnership trial = 63 hectares exempt
   → Funded hectares: 420 - 63 = 357

8. CALCULATE: 357 × $38,500 = $13,744,500
"""

DECOY_ANSWERS = """
DECOY ANSWERS (wrong paths that produce confident-sounding numbers):

$19,349,400 — Used revised drought tolerance (correct 6.6) but forgot it's < 7.0,
               or used ORIGINAL tolerance (7.3 ≥ 7.0 → Tier A). 357 × $54,200.

$16,170,000 — Correct tier (B) but forgot to exclude partnership hectares.
               420 × $38,500.

$22,764,000 — Wrong tier (A) AND forgot partnership exclusion.
               420 × $54,200.

$12,574,400 — Used wrong station (EA-03 instead of EA-07 after reclassification).
               EA-03: 290 hectares × 80% non-exempt × $54,200/ha (its cultivar CA-612
               has drought tolerance 7.8 ≥ 7.0 → Tier A).

$8,932,000  — Used EA-03's hectares (290) with Tier B rate. 232 × $38,500.
               Multiple errors combined.

"Cannot be determined" — Incorrect. All required information IS present in the
                          document; it just requires 8 reasoning hops across
                          7 widely separated paragraphs.
"""


def score_answer(response: str) -> dict:
    """Score a model's response against the correct answer."""
    import re

    # Extract dollar amounts from response
    amounts = re.findall(r'\$[\d,]+(?:\.\d+)?', response)
    amounts_clean = [int(a.replace('$', '').replace(',', '').split('.')[0]) for a in amounts]

    # Check for correct answer
    correct = CORRECT_ANSWER in amounts_clean

    # Check for specific decoy answers
    decoys_hit = []
    decoy_values = [19_349_400, 16_170_000, 22_764_000, 12_574_400, 8_932_000]
    for d in decoy_values:
        if d in amounts_clean:
            decoys_hit.append(d)

    # Check reasoning steps
    reasoning_markers = {
        "identified_reclassification": any(w in response.lower() for w in ["reclassif", "reassign", "moved to central"]),
        "used_ea07": "EA-07" in response or "ea-07" in response.lower(),
        "found_ma447": "MA-447" in response or "ma-447" in response.lower(),
        "used_revised_drought": "6.6" in response,
        "used_original_drought_WRONG": "7.3" in response and "revised" not in response.lower(),
        "identified_tier_b": "tier b" in response.lower() or "38,500" in response or "38500" in response,
        "excluded_partnership": any(w in response.lower() for w in ["exempt", "exclud", "partnership"]),
        "correct_funded_hectares": "357" in response,
    }

    return {
        "correct": correct,
        "answer_found": amounts_clean,
        "decoys_hit": decoys_hit,
        "reasoning_markers": reasoning_markers,
        "score": sum([
            correct * 50,  # final answer worth 50 points
            reasoning_markers["identified_reclassification"] * 10,
            reasoning_markers["used_ea07"] * 5,
            reasoning_markers["found_ma447"] * 5,
            reasoning_markers["used_revised_drought"] * 10,
            reasoning_markers["identified_tier_b"] * 5,
            reasoning_markers["excluded_partnership"] * 10,
            reasoning_markers["correct_funded_hectares"] * 5,
        ]),
        "max_score": 100,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate needle-in-haystack test")
    parser.add_argument("--target-tokens", type=int, default=200_000,
                        help="Target token count for haystack (default: 200000)")
    parser.add_argument("--output", type=str, default="haystack.txt",
                        help="Output file path (default: haystack.txt)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--question-file", type=str, default="question_answer.txt",
                        help="Output file for question/answer (default: question_answer.txt)")
    args = parser.parse_args()

    print(f"Generating haystack targeting ~{args.target_tokens:,} tokens...")
    result = generate_haystack(
        target_tokens=args.target_tokens,
        seed=args.seed,
        output_path=args.output,
    )

    print(f"  Paragraphs: {result['total_paragraphs']:,}")
    print(f"  Words:      {result['word_count']:,}")
    print(f"  Tokens:     ~{result['estimated_tokens']:,}")
    print(f"  Characters: {result['char_count']:,}")
    print(f"  File:       {result['output_path']}")
    print(f"\nNeedle positions (paragraph index):")
    for label, pos in result['needle_positions'].items():
        print(f"  {label}: paragraph {pos} / {result['total_paragraphs']}")
    print(f"\nCorrect answer: ${CORRECT_ANSWER:,}")

    # Write question/answer file
    qa_text = f"""QUESTION:
{QUESTION}

CORRECT ANSWER:
${CORRECT_ANSWER:,}

{REASONING_CHAIN}

{DECOY_ANSWERS}
"""
    Path(args.question_file).write_text(qa_text, encoding="utf-8")
    print(f"\nQuestion/answer written to: {args.question_file}")


if __name__ == "__main__":
    main()
