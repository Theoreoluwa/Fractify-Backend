REGION_SEVERITY = {
    "Wrist": "HIGH",
    "Radius": "HIGH",
    "Ulna": "MEDIUM",
    "MCP": "MEDIUM",
    "PIP": "LOW",
    "DIP": "LOW"
}

SEVERITY_PRIORITY = {
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1
}


def get_region_severity(region_class: str, classification: str, confidence: float) -> str:
    if classification != "fracture":
        return "NONE"

    # Handle "Fractured MCP" format by extracting the base region
    base_region = region_class
    if region_class.startswith("Fractured "):
        base_region = region_class.replace("Fractured ", "")

    base_severity = REGION_SEVERITY.get(base_region, "MEDIUM")

    if confidence < 0.6:
        priority = SEVERITY_PRIORITY.get(base_severity, 1)
        if priority > 1:
            priority -= 1
        for sev, pri in SEVERITY_PRIORITY.items():
            if pri == priority:
                return sev

    return base_severity


def get_overall_severity(prediction_results: list) -> str:
    highest_priority = 0

    for result in prediction_results:
        if result["classification"] == "fracture":
            severity = result["severity"]
            priority = SEVERITY_PRIORITY.get(severity, 0)
            if priority > highest_priority:
                highest_priority = priority

    for sev, pri in SEVERITY_PRIORITY.items():
        if pri == highest_priority:
            return sev

    return "NONE"