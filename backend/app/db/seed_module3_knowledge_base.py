"""
Module 3 — Environmental Knowledge Base
MILESTONE 2b: Content population — environmental_knowledge_base.

Every row below describes a VARIABLE, not a mine. No mine_code, no
mine-specific values, no statistical results from this platform's own
data — that separation is the entire point of this module (Knowledge Base
= generic scientific layer; mine-specific learning happens later in
Module 4/5).

knowledge_type classification
------------------------------
  established_principle  -> documented in meteorology / psychrometry /
                             coal science / mining engineering literature
  operational_assumption -> a reasonable, widely-used heuristic in coal
                             handling practice, but context-dependent
                             (e.g. depends on stockpile geometry, coal
                             rank, exposure)
  project_specific_rule  -> a composite/derived metric defined for this
                             platform; grounded in real physical drivers,
                             but the exact formula/weighting is a design
                             choice rather than a citable external result

confidence_level
------------------
  high   -> well-established scientific relationship
  medium -> supported but context-dependent
  low    -> hypothesis to be statistically validated by Module 4
            (Influence Quantification Engine)

scientific_references are citation ANCHORS (standard names, book titles,
named physical principles) — never reproduced text from any source.
"""

import sqlite3

# ---------------------------------------------------------------------------
# Common reference anchors reused across multiple variables
# ---------------------------------------------------------------------------
REF_PSYCHROMETRY = "Psychrometric principles (ASHRAE Handbook – Fundamentals, Psychrometrics chapter)"
REF_WMO_GLOSSARY = "American Meteorological Society, Glossary of Meteorology"
REF_PENMAN = "Penman (1948) evaporation theory; FAO Penman-Monteith reference evapotranspiration framework (FAO Irrigation and Drainage Paper 56)"
REF_COAL_HANDBOOK = "Osborne, D. (ed.), The Coal Handbook: Towards Cleaner Production, Vol. 1 — coal preparation and stockpile behaviour"
REF_SPEIGHT = "Speight, J.G., Handbook of Coal Analysis (2nd ed.) — coal moisture and quality relationships"
REF_ASTM_MOISTURE = "ASTM D3302/D3173 — Standard Test Methods for Total Moisture and Moisture in the Analysis Sample of Coal"
REF_IS1350 = "IS 1350 (Part I) — Indian Standard Methods of Test for Coal and Coke, Proximate Analysis"
REF_CIMFR_WEATHERING = "Central Institute of Mining and Fuel Research (CIMFR) literature on coal stockpile weathering and spontaneous moisture/oxidation behaviour"
REF_WMO_CODE_TABLE = "WMO Manual on Codes (WMO-No. 306) — present weather code definitions"

ROWS = [
    # ---------------- RAW WEATHER VARIABLES ----------------
    dict(
        variable_name="temperature",
        display_name="Temperature",
        variable_category="raw_weather",
        unit="°C",
        scientific_definition="Ambient air temperature at 2m above ground, reported as daily mean/max/min.",
        physical_meaning="Temperature governs the rate of evaporation from exposed coal surfaces and stockpiles: "
                          "warmer air can hold more water vapour and drives a steeper vapour-pressure gradient, "
                          "accelerating moisture loss when humidity is not also elevated.",
        operational_interpretation="Sustained high temperature, combined with low humidity, favours natural drying "
                                    "of coal before sampling. High temperature alongside high humidity does not "
                                    "guarantee drying — humidity must be read jointly with temperature.",
        source_table="weather_records",
        source_column="temperature_max_c,temperature_min_c,temperature_mean_c",
        knowledge_type="established_principle",
        scientific_references=f"{REF_PSYCHROMETRY}; {REF_PENMAN}",
        confidence_level="high",
        confidence_rationale="Temperature's role in evaporation rate is fundamental psychrometric physics, "
                              "independent of any specific mine or coal seam.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="relative_humidity",
        display_name="Relative Humidity",
        variable_category="raw_weather",
        unit="%",
        scientific_definition="Ratio of actual water vapour partial pressure in the air to the saturation vapour "
                               "pressure at the same temperature, expressed as a percentage.",
        physical_meaning="High relative humidity generally increases surface moisture retention because it "
                          "reduces the vapour-pressure gradient that drives evaporation from exposed coal — "
                          "the air is already closer to saturation, so it can absorb less additional moisture.",
        operational_interpretation="Periods of high relative humidity should be treated as conditions where "
                                    "natural drying of stockpiled or exposed coal is suppressed, raising the "
                                    "likelihood of elevated surface moisture at the next sampling event.",
        source_table="weather_records",
        source_column="relative_humidity_mean_pct,relative_humidity_max_pct,relative_humidity_min_pct",
        knowledge_type="established_principle",
        scientific_references=f"{REF_PSYCHROMETRY}; {REF_WMO_GLOSSARY}",
        confidence_level="high",
        confidence_rationale="The inverse relationship between relative humidity and evaporative drying is "
                              "fundamental atmospheric physics.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="rainfall",
        display_name="Rainfall",
        variable_category="raw_weather",
        unit="mm",
        scientific_definition="Total liquid precipitation accumulated over a 24-hour period.",
        physical_meaning="Direct precipitation deposits free water onto exposed coal and stockpile surfaces, "
                          "which can be absorbed into surface layers and pore structure, directly raising "
                          "surface and near-surface moisture content.",
        operational_interpretation="Rainfall on or shortly before a sampling date is one of the most direct and "
                                    "immediate environmental drivers of elevated moisture readings; recent "
                                    "rainfall should generally raise expectation of higher total moisture.",
        source_table="weather_records",
        source_column="rainfall_mm",
        knowledge_type="established_principle",
        scientific_references=f"{REF_COAL_HANDBOOK}; {REF_CIMFR_WEATHERING}",
        confidence_level="high",
        confidence_rationale="Direct wetting of exposed material by precipitation is a basic, well-documented "
                              "physical mechanism in coal handling literature.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="dew_point",
        display_name="Dew Point",
        variable_category="raw_weather",
        unit="°C",
        scientific_definition="The temperature to which air must be cooled (at constant pressure and moisture "
                               "content) to reach saturation, at which point condensation begins.",
        physical_meaning="When ambient/surface temperature approaches the dew point, condensation can form on "
                          "exposed coal and stockpile surfaces — an indirect moisture source distinct from "
                          "rainfall, often occurring overnight or in early morning.",
        operational_interpretation="A dew point close to the prevailing air temperature signals elevated "
                                    "condensation risk, particularly overnight; this can add surface moisture "
                                    "even on days with no recorded rainfall.",
        source_table="weather_records",
        source_column="dew_point_mean_c",
        knowledge_type="established_principle",
        scientific_references=f"{REF_PSYCHROMETRY}; {REF_WMO_GLOSSARY}",
        confidence_level="high",
        confidence_rationale="Dew point physics and condensation onset are standard, well-established "
                              "meteorological principles.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="wind_speed",
        display_name="Wind Speed",
        variable_category="raw_weather",
        unit="km/h",
        scientific_definition="Horizontal air movement speed measured at standard anemometer height, reported "
                               "as a daily representative (maximum) value.",
        physical_meaning="Wind accelerates evaporation by continuously replacing the boundary-layer air "
                          "immediately above a wet surface with drier air, sustaining the vapour-pressure "
                          "gradient that drives moisture loss. This effect is amplified when ambient humidity "
                          "is low, and suppressed when humidity is already high.",
        operational_interpretation="Higher wind speed generally promotes faster drying of exposed coal surfaces, "
                                    "but its benefit depends on simultaneous humidity — strong wind in already "
                                    "humid air has a much weaker drying effect than the same wind speed in dry air.",
        source_table="weather_records",
        source_column="wind_speed_mean_kmh",
        knowledge_type="established_principle",
        scientific_references=f"{REF_PENMAN}; {REF_PSYCHROMETRY}",
        confidence_level="medium",
        confidence_rationale="The boundary-layer mechanism is well established, but its practical drying impact "
                              "is explicitly context-dependent on humidity, justifying medium rather than high "
                              "confidence for the variable in isolation.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="wind_gust",
        display_name="Wind Gust",
        variable_category="raw_weather",
        unit="km/h",
        scientific_definition="Maximum short-duration wind speed peak recorded during the day, exceeding the "
                               "sustained wind speed.",
        physical_meaning="Gusts produce brief, intense boundary-layer disruption that can transiently accelerate "
                          "surface drying and may also entrain fine coal particles, but their short duration "
                          "limits cumulative drying contribution compared to sustained wind.",
        operational_interpretation="Wind gust is best read as a secondary/contextual indicator alongside "
                                    "sustained wind speed rather than a standalone driver of moisture or quality "
                                    "outcomes.",
        source_table="weather_records",
        source_column="wind_gust_max_kmh",
        knowledge_type="operational_assumption",
        scientific_references=f"{REF_WMO_GLOSSARY}",
        confidence_level="low",
        confidence_rationale="The transient nature of gusts makes their incremental contribution to daily-scale "
                              "moisture/quality outcomes plausible but not well quantified in literature; "
                              "treated as a hypothesis pending statistical validation.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="surface_pressure",
        display_name="Surface Pressure",
        variable_category="raw_weather",
        unit="hPa",
        scientific_definition="Atmospheric pressure at ground/station level.",
        physical_meaning="Surface pressure itself does not directly alter coal moisture or quality, but "
                          "pressure trends are a standard indicator of approaching weather systems (e.g. "
                          "falling pressure often precedes precipitation), making it an indirect early-warning "
                          "signal rather than a direct physical driver.",
        operational_interpretation="Use surface pressure trend as a leading indicator of incoming weather change "
                                    "rather than as a direct explanatory variable for moisture or GCV outcomes.",
        source_table="weather_records",
        source_column="surface_pressure_mean_hpa",
        knowledge_type="established_principle",
        scientific_references=f"{REF_WMO_GLOSSARY}",
        confidence_level="low",
        confidence_rationale="Pressure's link to coal quality outcomes is indirect (via its association with "
                              "other variables like rainfall), so any direct influence claim is weak by nature "
                              "and should be statistically tested rather than assumed.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="cloud_cover",
        display_name="Cloud Cover",
        variable_category="raw_weather",
        unit="%",
        scientific_definition="Fraction of sky covered by clouds, expressed as a daily mean percentage.",
        physical_meaning="Cloud cover attenuates incoming shortwave solar radiation, reducing the energy "
                          "available to drive evaporation at the surface, and is also frequently associated "
                          "with higher humidity and precipitation likelihood.",
        operational_interpretation="Persistently high cloud cover should be read as suppressing natural drying "
                                    "potential, independent of whether rain actually falls.",
        source_table="weather_records",
        source_column="cloud_cover_mean_pct",
        knowledge_type="established_principle",
        scientific_references=f"{REF_PENMAN}; {REF_WMO_GLOSSARY}",
        confidence_level="high",
        confidence_rationale="The radiative attenuation effect of cloud cover on surface evaporation is "
                              "standard atmospheric science.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="visibility",
        display_name="Visibility",
        variable_category="raw_weather",
        unit="km",
        scientific_definition="Horizontal distance at which an object or light can be clearly discerned, "
                               "reduced by fog, haze, dust, or precipitation.",
        physical_meaning="Visibility is a composite proxy for atmospheric moisture and particulate content "
                          "(fog/mist reduce visibility and indicate near-saturation air; dust/haze reduce "
                          "visibility without indicating moisture) — it does not have a single, unambiguous "
                          "physical mechanism linking it to coal quality.",
        operational_interpretation="Low visibility should prompt a check of the specific cause (fog vs. dust vs. "
                                    "rain) via the other weather variables before drawing any moisture or "
                                    "quality conclusion — visibility alone is not a reliable standalone driver.",
        source_table="weather_records",
        source_column="visibility_mean_km",
        knowledge_type="operational_assumption",
        scientific_references=f"{REF_WMO_GLOSSARY}",
        confidence_level="low",
        confidence_rationale="Visibility is an ambiguous composite signal with multiple possible causes, so its "
                              "standalone explanatory value for coal quality is weak and unproven.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="solar_radiation",
        display_name="Solar Radiation",
        variable_category="raw_weather",
        unit="MJ/m²",
        scientific_definition="Total downward shortwave solar radiation received at the surface over the day "
                               "(daily radiation sum).",
        physical_meaning="Solar radiation supplies the thermal energy that drives evaporation directly at the "
                          "surface — it is the primary energy input in the surface energy balance underlying "
                          "evapotranspiration theory.",
        operational_interpretation="Higher solar radiation promotes drying of exposed coal and stockpile "
                                    "surfaces; this effect is most reliable when combined with low cloud cover "
                                    "and moderate-to-low humidity.",
        source_table="weather_records",
        source_column="solar_radiation_mj_m2",
        knowledge_type="established_principle",
        scientific_references=f"{REF_PENMAN}",
        confidence_level="high",
        confidence_rationale="Solar radiation as the primary energy source for evaporation is foundational to "
                              "established evapotranspiration theory (Penman-Monteith framework).",
        requires_statistical_validation=1,
    ),

    # ---------------- SAME-DAY DERIVED FEATURES ----------------
    dict(
        variable_name="thermal_stress",
        display_name="Thermal Stress Index",
        variable_category="same_day_derived",
        unit="score (0-100)",
        scientific_definition="A composite same-day index combining excess temperature above a 25°C baseline "
                               "with relative humidity, intended to approximate combined heat+moisture load on "
                               "exposed coal surfaces (computed in environmental_features.thermal_stress_index).",
        physical_meaning="Captures conditions where high temperature and high humidity occur together — a "
                          "combination that limits the cooling/drying benefit of heat alone, analogous to how "
                          "heat-index concepts in human comfort science combine temperature and humidity.",
        operational_interpretation="A high thermal stress score indicates hot-and-humid conditions where the "
                                    "expected drying benefit of high temperature is being undermined by "
                                    "simultaneously high humidity — do not assume high temperature alone implies "
                                    "drying without checking this index.",
        source_table="environmental_features",
        source_column="thermal_stress_index",
        knowledge_type="project_specific_rule",
        scientific_references=f"{REF_PSYCHROMETRY} (heat-index style combined-variable reasoning, adapted)",
        confidence_level="medium",
        confidence_rationale="The temperature+humidity interaction this index captures is a recognised physical "
                              "phenomenon, but the specific formula/weighting used here is a project design "
                              "choice rather than a directly citable external index.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="dew_spread",
        display_name="Dew Spread",
        variable_category="same_day_derived",
        unit="°C",
        scientific_definition="Difference between mean air temperature and mean dew point on the same day "
                               "(temperature_mean_c − dew_point_mean_c).",
        physical_meaning="A small or negative dew spread means the air is close to saturation, making "
                          "condensation onto cool surfaces (including coal stockpiles) more likely, "
                          "particularly as surfaces cool overnight below the air temperature.",
        operational_interpretation="A narrow dew spread is an early-warning signal for condensation-driven "
                                    "moisture gain that would not be captured by rainfall data alone.",
        source_table="environmental_features",
        source_column="dew_spread_c",
        knowledge_type="established_principle",
        scientific_references=f"{REF_PSYCHROMETRY}; {REF_WMO_GLOSSARY}",
        confidence_level="high",
        confidence_rationale="Dew spread as a saturation/condensation-risk indicator is a direct, well-"
                              "established consequence of psychrometric theory.",
        requires_statistical_validation=1,
    ),

    # ---------------- ROLLING / WINDOW DERIVED FEATURES ----------------
    dict(
        variable_name="drying_potential",
        display_name="Drying Potential",
        variable_category="rolling_derived",
        unit="score (0-100)",
        scientific_definition="A composite index combining temperature, wind speed, and solar radiation "
                               "(favourable to drying) against humidity and rainfall (unfavourable to drying), "
                               "rescaled to a 0-100 scale (derived_environmental_features.drying_potential).",
        physical_meaning="Approximates the net surface energy/vapour-gradient balance that determines whether "
                          "exposed coal is likely to lose or retain moisture under prevailing conditions, "
                          "drawing on the same variables used in standard evapotranspiration theory.",
        operational_interpretation="High drying potential suggests conditions favourable to natural moisture "
                                    "reduction before sampling; low or negative drying potential suggests "
                                    "conditions that will tend to preserve or increase existing moisture.",
        source_table="derived_environmental_features",
        source_column="drying_potential",
        knowledge_type="project_specific_rule",
        scientific_references=f"{REF_PENMAN} (conceptual basis for combining the underlying variables)",
        confidence_level="medium",
        confidence_rationale="Each input variable's individual role in drying is well established, but the "
                              "specific composite formula and coefficients are a project design choice not "
                              "found verbatim in literature, so the index as a whole warrants medium confidence "
                              "pending statistical validation.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="environmental_risk_index",
        display_name="Environmental Risk Index",
        variable_category="rolling_derived",
        unit="score (0-100)",
        scientific_definition="A composite index, higher values indicating conditions more adverse to coal "
                               "quality, derived as the inverse of drying potential adjusted by trailing 7-day "
                               "rainfall accumulation (derived_environmental_features.environmental_risk_index).",
        physical_meaning="Intended as a single summary signal of how unfavourable recent and current weather "
                          "has been for maintaining low moisture / high GCV coal, by combining a same-day "
                          "drying assessment with a short-term wetness memory.",
        operational_interpretation="Use as a quick at-a-glance risk flag (e.g. for the Overview dashboard / "
                                    "Alerts), not as a substitute for inspecting the underlying drivers — the "
                                    "Dominant Driver / Explainable AI layer should always accompany this score.",
        source_table="derived_environmental_features",
        source_column="environmental_risk_index",
        knowledge_type="project_specific_rule",
        scientific_references=None,
        confidence_level="low",
        confidence_rationale="This is a project-defined composite with no direct external literature "
                              "equivalent; its predictive validity must be established empirically by Module 4 "
                              "before it is treated as more than a heuristic summary.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="weather_stability_index",
        display_name="Weather Stability Index",
        variable_category="rolling_derived",
        unit="score (0-100)",
        scientific_definition="A composite index, higher values indicating more consistent recent weather, "
                               "computed from the inverse of normalised 7-day standard deviation of temperature "
                               "and rainfall (derived_environmental_features.weather_stability_index).",
        physical_meaning="Volatile, rapidly-changing weather (e.g. alternating wet/dry or hot/cold spells) "
                          "can produce inconsistent moisture conditions across a stockpile or sampling period, "
                          "whereas stable weather tends to produce more uniform, predictable conditions.",
        operational_interpretation="Low stability suggests sampling results during this period may be more "
                                    "variable or less representative of 'typical' conditions for the mine; "
                                    "high stability supports more confidence in a single sample being "
                                    "representative.",
        source_table="derived_environmental_features",
        source_column="weather_stability_index",
        knowledge_type="project_specific_rule",
        scientific_references=None,
        confidence_level="low",
        confidence_rationale="The underlying intuition (variability reduces representativeness) is reasonable "
                              "but the specific index construction is project-specific and not independently "
                              "validated against this platform's sampling data yet.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="consecutive_wet_days",
        display_name="Consecutive Wet Days",
        variable_category="rolling_derived",
        unit="days",
        scientific_definition="Count of consecutive days (ending on the current date) with rainfall exceeding "
                               "a 1.0mm threshold (derived_environmental_features.consecutive_wet_days).",
        physical_meaning="Multi-day wet spells allow progressively deeper water penetration into stockpiles "
                          "and exposed coal, a cumulative wetting effect that a single day's rainfall total "
                          "does not capture.",
        operational_interpretation="Longer wet runs should raise expectation of higher accumulated moisture "
                                    "more than an equivalent single-day total spread across a drier period "
                                    "would.",
        source_table="derived_environmental_features",
        source_column="consecutive_wet_days",
        knowledge_type="operational_assumption",
        scientific_references=f"{REF_COAL_HANDBOOK}; {REF_CIMFR_WEATHERING}",
        confidence_level="medium",
        confidence_rationale="Cumulative wetting effects of multi-day rain spells on stockpiled bulk materials "
                              "are a recognised operational concern in coal handling practice, though the exact "
                              "1.0mm wet-day threshold used here is a project-chosen convention.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="consecutive_dry_days",
        display_name="Consecutive Dry Days",
        variable_category="rolling_derived",
        unit="days",
        scientific_definition="Count of consecutive days (ending on the current date) with rainfall at or below "
                               "a 1.0mm threshold (derived_environmental_features.consecutive_dry_days).",
        physical_meaning="Extended dry spells allow progressive, cumulative surface and near-surface drying, "
                          "analogous to soil-moisture depletion concepts in agricultural meteorology.",
        operational_interpretation="Longer dry runs should raise expectation of lower moisture / potentially "
                                    "higher GCV at the next sampling event, especially when combined with high "
                                    "drying potential.",
        source_table="derived_environmental_features",
        source_column="consecutive_dry_days",
        knowledge_type="operational_assumption",
        scientific_references=f"{REF_COAL_HANDBOOK}",
        confidence_level="medium",
        confidence_rationale="Cumulative drying over consecutive dry days is a reasonable, widely-assumed "
                              "operational heuristic, but its precise quantitative effect on coal moisture is "
                              "context-dependent (stockpile size/shape, coal rank) and not independently proven "
                              "here.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="moisture_accumulation_index",
        display_name="Moisture Accumulation Index",
        variable_category="rolling_derived",
        unit="score (unitless, decayed cumulative)",
        scientific_definition="An exponentially-decayed cumulative signal combining daily rainfall and humidity, "
                               "with a decay factor of 0.85 chosen to give roughly a 7-day half-life "
                               "(derived_environmental_features.moisture_accumulation_index).",
        physical_meaning="Models moisture as a 'reservoir' that fills with rain/humidity and drains gradually "
                          "over time, rather than treating each day in isolation — analogous to soil-moisture "
                          "accounting models used in hydrology.",
        operational_interpretation="A persistently rising index across several days indicates accumulating "
                                    "moisture stress even without any single extreme rainfall event; a falling "
                                    "index indicates the stockpile/exposed coal is in a net-drying phase.",
        source_table="derived_environmental_features",
        source_column="moisture_accumulation_index",
        knowledge_type="project_specific_rule",
        scientific_references="General hydrological soil-moisture accounting / antecedent precipitation index "
                               "concepts (conceptual basis only — exact decay constant is project-specific)",
        confidence_level="low",
        confidence_rationale="The decay-based accumulation concept is borrowed from a well-known class of "
                              "hydrological models, but the specific 0.85 decay constant and weighting used "
                              "here are project choices, not independently validated against this platform's "
                              "sampling outcomes.",
        requires_statistical_validation=1,
    ),
    dict(
        variable_name="temperature_range",
        display_name="Temperature Range",
        variable_category="same_day_derived",
        unit="°C",
        scientific_definition="Difference between daily maximum and minimum temperature "
                               "(environmental_features.temperature_range_c).",
        physical_meaning="A large diurnal temperature range is typically associated with clear skies and low "
                          "humidity (since cloud cover and moisture buffer both daytime heating and nighttime "
                          "cooling), making it an indirect proxy for generally dry, drying-favourable conditions.",
        operational_interpretation="A wide temperature range on a given day is a secondary signal supporting "
                                    "(but not proving on its own) an assessment of favourable drying conditions "
                                    "— it should be read alongside cloud cover and humidity, not in isolation.",
        source_table="environmental_features",
        source_column="temperature_range_c",
        knowledge_type="operational_assumption",
        scientific_references=f"{REF_WMO_GLOSSARY}",
        confidence_level="low",
        confidence_rationale="The clear-sky/diurnal-range association is a recognised meteorological pattern, "
                              "but its use here as an indirect coal-drying proxy is an inferential step removed "
                              "from direct causation, warranting low confidence as a standalone variable.",
        requires_statistical_validation=1,
    ),
]


def populate(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cols = [
            "variable_name", "display_name", "variable_category", "unit",
            "scientific_definition", "physical_meaning", "operational_interpretation",
            "source_table", "source_column", "knowledge_type", "scientific_references",
            "confidence_level", "confidence_rationale", "requires_statistical_validation",
        ]
        sql = f"""
            INSERT INTO environmental_knowledge_base ({",".join(cols)})
            VALUES ({",".join(["?"] * len(cols))})
            ON CONFLICT(variable_name) DO UPDATE SET
                display_name=excluded.display_name,
                variable_category=excluded.variable_category,
                unit=excluded.unit,
                scientific_definition=excluded.scientific_definition,
                physical_meaning=excluded.physical_meaning,
                operational_interpretation=excluded.operational_interpretation,
                source_table=excluded.source_table,
                source_column=excluded.source_column,
                knowledge_type=excluded.knowledge_type,
                scientific_references=excluded.scientific_references,
                confidence_level=excluded.confidence_level,
                confidence_rationale=excluded.confidence_rationale,
                requires_statistical_validation=excluded.requires_statistical_validation,
                updated_at=datetime('now')
        """
        for row in ROWS:
            cur.execute(sql, [row.get(c) for c in cols])
        con.commit()
        return len(ROWS)
    finally:
        con.close()


if __name__ == "__main__":
    import sys
    db_file = sys.argv[1] if len(sys.argv) > 1 else "coal_eie.db"
    n = populate(db_file)
    print(f"Populated/updated {n} environmental_knowledge_base rows in {db_file}")
