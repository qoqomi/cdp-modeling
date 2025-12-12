"""
CDP Module 2 Response Schema Definition
CDP 실제 UI 기반 응답 형식 정의

각 질문별로 테이블 구조, 컬럼 타입, 옵션 등을 정확하게 정의
"""

from typing import Dict, List, Any, Optional
from enum import Enum


class FieldType(str, Enum):
    SELECT = "select"                 # 단일 선택 드롭다운
    MULTISELECT = "multiselect"       # 다중 선택 체크박스
    TEXT = "text"                     # 짧은 텍스트 (숫자 포함)
    TEXTAREA = "textarea"             # 긴 서술형
    NUMBER = "number"                 # 숫자 입력
    PERCENTAGE = "percentage"         # 퍼센트 입력
    YEAR = "year"                     # 연도 입력
    GROUPED_SELECT = "grouped_select" # 그룹화된 드롭다운


# ============================================================================
# 공통 옵션 정의
# ============================================================================

COMMON_OPTIONS = {
    "yes_no": ["Yes", "No"],

    "yes_no_plan": [
        "Yes",
        "No, but we plan to within the next two years",
        "No, and we do not plan to within the next two years"
    ],

    "process_in_place": [
        "Yes",
        "No – this is our first year of disclosure",
        "No – we have previously disclosed but have not had a process",
        "No – we do not have a process"
    ],

    "dependencies_impacts": [
        "Both dependencies and impacts",
        "Dependencies only",
        "Impacts only"
    ],

    "risks_opportunities": [
        "Both risks and opportunities",
        "Risks only",
        "Opportunities only"
    ],

    "environmental_issues": [
        "Climate change",
        "Forests",
        "Water",
        "Plastics",
        "Biodiversity"
    ],

    "diro_covered": [  # Dependencies, Impacts, Risks, Opportunities
        "Dependencies",
        "Impacts",
        "Risks",
        "Opportunities"
    ],

    "value_chain_stages": [
        "Direct operations",
        "Upstream value chain",
        "Downstream value chain"
    ],

    "coverage": ["Full", "Partial"],

    "supplier_tiers": [
        "Tier 1 suppliers",
        "Tier 2 suppliers",
        "Tier 3 suppliers",
        "Tier 4+ suppliers"
    ],

    "assessment_type": [
        "Qualitative only",
        "Quantitative only",
        "Qualitative and quantitative"
    ],

    "frequency": [
        "More than once a year",
        "Annually",
        "Every two years",
        "Every three years or more",
        "As important matters arise",
        "Not defined"
    ],

    "time_horizons": [
        "Short-term",
        "Medium-term",
        "Long-term"
    ],

    "location_specificity": [
        "Site-specific",
        "Local",
        "Sub-national",
        "National",
        "Not location specific"
    ],

    "integration": [
        "Integrated into multi-disciplinary organization-wide risk management process",
        "A specific environmental risk management process"
    ],

    "primary_reasons_no_process": [
        "Lack of internal resources, capabilities, or expertise",
        "No standardized procedure",
        "Not an immediate strategic priority",
        "Judged to be unimportant or not relevant",
        "Other, please specify"
    ],

    "significance_determination": [
        "Size/scale of impact",
        "Likelihood of occurrence",
        "Potential financial impact",
        "Stakeholder concern",
        "Regulatory requirements",
        "Other, please specify"
    ]
}


# 그룹화된 옵션 (Grouped Select용)
GROUPED_OPTIONS = {
    "risk_types": {
        "Acute physical": [
            "Avalanche",
            "Cold wave/frost",
            "Cyclone, hurricane, typhoon",
            "Drought",
            "Flood (coastal, fluvial, pluvial, ground water)",
            "Heat wave",
            "Heavy precipitation (rain, hail, snow/ice)",
            "Landslide",
            "Storm (including blizzards, dust, and sandstorms)",
            "Tornado",
            "Wildfire",
            "Other acute physical, please specify"
        ],
        "Chronic physical": [
            "Changing precipitation patterns and types",
            "Changing temperature (air, freshwater, marine water)",
            "Coastal erosion",
            "Heat stress",
            "Ocean acidification",
            "Permafrost thawing",
            "Sea level rise",
            "Soil degradation",
            "Soil erosion",
            "Water scarcity",
            "Water stress",
            "Other chronic physical, please specify"
        ],
        "Policy": [
            "Carbon pricing mechanisms",
            "Enhanced emissions-reporting obligations",
            "Mandates on and regulation of existing products and services",
            "Other policy, please specify"
        ],
        "Technology": [
            "Substitution of existing products and services with lower emissions options",
            "Unsuccessful investment in new technologies",
            "Other technology, please specify"
        ],
        "Market": [
            "Changing customer behavior",
            "Increased cost of raw materials",
            "Uncertainty in market signals",
            "Other market, please specify"
        ],
        "Reputation": [
            "Increased stakeholder concern or negative stakeholder feedback",
            "Stigmatization of sector",
            "Other reputation, please specify"
        ],
        "Liability": [
            "Exposure to litigation",
            "Non-compliance with regulations",
            "Other liability, please specify"
        ]
    },

    "opportunity_types": {
        "Resource efficiency": [
            "Use of more efficient production and distribution processes",
            "Use of recycling",
            "Move to more efficient buildings",
            "Reduced water usage and consumption",
            "Other resource efficiency, please specify"
        ],
        "Energy source": [
            "Use of lower-emission sources of energy",
            "Use of supportive policy incentives",
            "Use of new technologies",
            "Participation in carbon market",
            "Other energy source, please specify"
        ],
        "Products and services": [
            "Development and/or expansion of low emission goods and services",
            "Development of climate adaptation solutions",
            "Shift in consumer preferences",
            "Other products and services, please specify"
        ],
        "Markets": [
            "Access to new markets",
            "Use of public-sector incentives",
            "Other markets, please specify"
        ],
        "Resilience": [
            "Resource substitutes/diversification",
            "Other resilience, please specify"
        ]
    },

    "tools_and_methods": {
        "Enterprise Risk Management": [
            "COSO Enterprise Risk Management Framework",
            "ISO 31000 Risk Management Standard",
            "Other enterprise risk management, please specify"
        ],
        "International methodologies": [
            "Environmental Impact Assessment",
            "IPCC Climate Change Projections",
            "ISO 14001 Environmental Management Standard",
            "Life Cycle Assessment",
            "Science Based Targets Initiative (SBTi)",
            "TCFD recommendations",
            "TNFD recommendations",
            "Other international methodology, please specify"
        ],
        "Sector-specific tools": [
            "SASB standards",
            "Other sector-specific, please specify"
        ],
        "Other tools": [
            "Internal company methods",
            "External consultants",
            "Scenario analysis",
            "Other, please specify"
        ]
    }
}


# ============================================================================
# 질문별 Response Schema 정의
# ============================================================================

RESPONSE_SCHEMAS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------------
    # 2.1 Time Horizons
    # ------------------------------------------------------------------------
    "2.1": {
        "title": "Time horizons definition",
        "response_type": "table",
        "allow_multiple_rows": True,
        "row_labels": ["Short-term", "Medium-term", "Long-term"],
        "columns": [
            {
                "id": "time_horizon",
                "header": "Time horizon",
                "type": "select",
                "options": ["Short-term", "Medium-term", "Long-term"],
                "required": True
            },
            {
                "id": "from_years",
                "header": "From (years)",
                "type": "number",
                "min_value": 0,
                "max_value": 100,
                "required": True
            },
            {
                "id": "to_years",
                "header": "To (years)",
                "type": "number",
                "min_value": 0,
                "max_value": 100,
                "required": True
            },
            {
                "id": "rationale",
                "header": "How this time horizon is linked to strategic and/or financial planning",
                "type": "textarea",
                "max_length": 1500,
                "required": False
            }
        ]
    },

    # ------------------------------------------------------------------------
    # 2.2 Dependencies and Impacts Process
    # ------------------------------------------------------------------------
    "2.2": {
        "title": "Environmental dependencies and impacts management process",
        "response_type": "table",
        "allow_multiple_rows": True,
        "columns": [
            {
                "id": "process_in_place",
                "header": "Process in place",
                "type": "select",
                "options": COMMON_OPTIONS["process_in_place"],
                "required": True
            },
            {
                "id": "deps_impacts_evaluated",
                "header": "Dependencies and/or impacts evaluated in this process",
                "type": "select",
                "options": COMMON_OPTIONS["dependencies_impacts"],
                "condition": {"process_in_place": "Yes"}
            },
            {
                "id": "primary_reason_no",
                "header": "Primary reason for not evaluating",
                "type": "select",
                "options": COMMON_OPTIONS["primary_reasons_no_process"],
                "condition": {"process_in_place": ["No – this is our first year of disclosure",
                                                    "No – we have previously disclosed but have not had a process",
                                                    "No – we do not have a process"]}
            },
            {
                "id": "explain_no_process",
                "header": "Explain why you do not evaluate and describe any plans",
                "type": "textarea",
                "max_length": 2500,
                "condition": {"process_in_place": ["No – this is our first year of disclosure",
                                                    "No – we have previously disclosed but have not had a process",
                                                    "No – we do not have a process"]}
            }
        ]
    },

    # ------------------------------------------------------------------------
    # 2.2.1 Risks and Opportunities Process
    # ------------------------------------------------------------------------
    "2.2.1": {
        "title": "Environmental risks and opportunities management process",
        "response_type": "table",
        "allow_multiple_rows": True,
        "columns": [
            {
                "id": "process_in_place",
                "header": "Process in place",
                "type": "select",
                "options": COMMON_OPTIONS["process_in_place"],
                "required": True
            },
            {
                "id": "risks_opps_evaluated",
                "header": "Risks and/or opportunities evaluated in this process",
                "type": "select",
                "options": COMMON_OPTIONS["risks_opportunities"],
                "condition": {"process_in_place": "Yes"}
            },
            {
                "id": "informed_by_deps_impacts",
                "header": "Is this process informed by the dependencies and/or impacts process?",
                "type": "select",
                "options": COMMON_OPTIONS["yes_no"],
                "condition": {"process_in_place": "Yes"}
            }
        ]
    },

    # ------------------------------------------------------------------------
    # 2.2.2 Process Details
    # ------------------------------------------------------------------------
    "2.2.2": {
        "title": "Details of identification, assessment, and management process",
        "response_type": "table",
        "allow_multiple_rows": True,
        "columns": [
            {
                "id": "environmental_issue",
                "header": "Environmental issue",
                "type": "multiselect",
                "options": COMMON_OPTIONS["environmental_issues"],
                "required": True
            },
            {
                "id": "diro_covered",
                "header": "Indicate which of dependencies, impacts, risks, and opportunities are covered",
                "type": "multiselect",
                "options": COMMON_OPTIONS["diro_covered"],
                "required": True
            },
            {
                "id": "value_chain_stages",
                "header": "Value chain stages covered",
                "type": "multiselect",
                "options": COMMON_OPTIONS["value_chain_stages"],
                "required": True
            },
            {
                "id": "coverage",
                "header": "Coverage",
                "type": "select",
                "options": COMMON_OPTIONS["coverage"],
                "required": True
            },
            {
                "id": "supplier_tiers",
                "header": "Supplier tiers covered",
                "type": "multiselect",
                "options": COMMON_OPTIONS["supplier_tiers"],
                "condition": {"value_chain_stages": "Upstream value chain"}
            },
            {
                "id": "assessment_type",
                "header": "Type of assessment",
                "type": "select",
                "options": COMMON_OPTIONS["assessment_type"],
                "required": True
            },
            {
                "id": "frequency",
                "header": "Frequency of assessment",
                "type": "select",
                "options": COMMON_OPTIONS["frequency"],
                "required": True
            },
            {
                "id": "time_horizons",
                "header": "Time horizons covered",
                "type": "multiselect",
                "options": COMMON_OPTIONS["time_horizons"],
                "required": True
            },
            {
                "id": "integration",
                "header": "Integration of risk management process",
                "type": "select",
                "options": COMMON_OPTIONS["integration"],
                "condition": {"diro_covered": "Risks"}
            },
            {
                "id": "location_specificity",
                "header": "Location-specificity used",
                "type": "multiselect",
                "options": COMMON_OPTIONS["location_specificity"]
            },
            {
                "id": "tools_used",
                "header": "Tools and/or methods used",
                "type": "grouped_select",
                "grouped_options": GROUPED_OPTIONS["tools_and_methods"]
            },
            {
                "id": "risk_types_considered",
                "header": "Risk types considered",
                "type": "grouped_select",
                "grouped_options": GROUPED_OPTIONS["risk_types"],
                "condition": {"diro_covered": "Risks"}
            },
            {
                "id": "opportunity_types_considered",
                "header": "Opportunity types considered",
                "type": "grouped_select",
                "grouped_options": GROUPED_OPTIONS["opportunity_types"],
                "condition": {"diro_covered": "Opportunities"}
            },
            {
                "id": "significance_determination",
                "header": "How significance is determined",
                "type": "multiselect",
                "options": COMMON_OPTIONS["significance_determination"]
            },
            {
                "id": "further_details",
                "header": "Further details of process",
                "type": "textarea",
                "max_length": 3500
            }
        ]
    },

    # ------------------------------------------------------------------------
    # 2.2.3 Identified Dependencies and Impacts
    # ------------------------------------------------------------------------
    "2.2.3": {
        "title": "Identified environmental dependencies and impacts",
        "response_type": "table",
        "allow_multiple_rows": True,
        "columns": [
            {
                "id": "environmental_issue",
                "header": "Environmental issue",
                "type": "select",
                "options": COMMON_OPTIONS["environmental_issues"],
                "required": True
            },
            {
                "id": "dependency_or_impact",
                "header": "Dependency or impact",
                "type": "select",
                "options": ["Dependency", "Impact"],
                "required": True
            },
            {
                "id": "type_of_dependency_impact",
                "header": "Type of dependency/impact",
                "type": "textarea",
                "max_length": 500,
                "required": True
            },
            {
                "id": "description",
                "header": "Description",
                "type": "textarea",
                "max_length": 2500
            },
            {
                "id": "value_chain_stage",
                "header": "Value chain stage",
                "type": "multiselect",
                "options": COMMON_OPTIONS["value_chain_stages"]
            }
        ]
    },

    # ------------------------------------------------------------------------
    # 2.2.4 Financial Services - Portfolio Analysis
    # ------------------------------------------------------------------------
    "2.2.4": {
        "title": "Portfolio environmental dependencies and impacts (Financial Services)",
        "response_type": "table",
        "allow_multiple_rows": True,
        "sector_specific": "Financial Services",
        "columns": [
            {
                "id": "process_in_place",
                "header": "Process in place",
                "type": "select",
                "options": COMMON_OPTIONS["yes_no_plan"],
                "required": True
            },
            {
                "id": "portfolio_coverage",
                "header": "Portfolio coverage",
                "type": "percentage",
                "min_value": 0,
                "max_value": 100
            },
            {
                "id": "methodology",
                "header": "Methodology used",
                "type": "textarea",
                "max_length": 2500
            }
        ]
    },

    # ------------------------------------------------------------------------
    # 2.2.5 Financial Services - Portfolio Risks/Opportunities
    # ------------------------------------------------------------------------
    "2.2.5": {
        "title": "Portfolio environmental risks and opportunities (Financial Services)",
        "response_type": "table",
        "allow_multiple_rows": True,
        "sector_specific": "Financial Services",
        "columns": [
            {
                "id": "process_in_place",
                "header": "Process in place",
                "type": "select",
                "options": COMMON_OPTIONS["yes_no_plan"],
                "required": True
            },
            {
                "id": "portfolio_coverage",
                "header": "Portfolio coverage",
                "type": "percentage",
                "min_value": 0,
                "max_value": 100
            },
            {
                "id": "risk_types",
                "header": "Risk types assessed",
                "type": "grouped_select",
                "grouped_options": GROUPED_OPTIONS["risk_types"]
            },
            {
                "id": "methodology",
                "header": "Methodology used",
                "type": "textarea",
                "max_length": 2500
            }
        ]
    },

    # ------------------------------------------------------------------------
    # 2.2.6 Scenario Analysis Usage
    # ------------------------------------------------------------------------
    "2.2.6": {
        "title": "Scenario analysis usage",
        "response_type": "table",
        "allow_multiple_rows": False,
        "columns": [
            {
                "id": "uses_scenario_analysis",
                "header": "Does your organization use scenario analysis?",
                "type": "select",
                "options": COMMON_OPTIONS["yes_no_plan"],
                "required": True
            },
            {
                "id": "explanation_no",
                "header": "Explain why not and any future plans",
                "type": "textarea",
                "max_length": 2500,
                "condition": {"uses_scenario_analysis": ["No, but we plan to within the next two years",
                                                          "No, and we do not plan to within the next two years"]}
            }
        ]
    },

    # ------------------------------------------------------------------------
    # 2.2.7 Scenario Analysis Details
    # ------------------------------------------------------------------------
    "2.2.7": {
        "title": "Scenario analysis details",
        "response_type": "table",
        "allow_multiple_rows": True,
        "columns": [
            {
                "id": "environmental_issue",
                "header": "Environmental issue",
                "type": "multiselect",
                "options": COMMON_OPTIONS["environmental_issues"],
                "required": True
            },
            {
                "id": "scenario_type",
                "header": "Type of scenario",
                "type": "select",
                "options": [
                    "Physical scenarios",
                    "Transition scenarios",
                    "Both physical and transition"
                ],
                "required": True
            },
            {
                "id": "scenario_source",
                "header": "Scenario source",
                "type": "select",
                "options": [
                    "IEA NZE 2050",
                    "IEA STEPS",
                    "IEA APS",
                    "IPCC RCP 2.6",
                    "IPCC RCP 4.5",
                    "IPCC RCP 8.5",
                    "NGFS scenarios",
                    "Internally developed",
                    "Other, please specify"
                ]
            },
            {
                "id": "time_horizon",
                "header": "Time horizon covered",
                "type": "multiselect",
                "options": COMMON_OPTIONS["time_horizons"]
            },
            {
                "id": "temperature_alignment",
                "header": "Temperature alignment",
                "type": "select",
                "options": ["1.5°C", "Below 2°C", "2°C", "Above 2°C", "N/A"]
            },
            {
                "id": "description",
                "header": "Description of scenario and results",
                "type": "textarea",
                "max_length": 3500
            }
        ]
    },

    # ------------------------------------------------------------------------
    # 2.2.8 Business Strategy Impact
    # ------------------------------------------------------------------------
    "2.2.8": {
        "title": "Business strategy impact from scenario analysis",
        "response_type": "table",
        "allow_multiple_rows": True,
        "columns": [
            {
                "id": "strategy_impacted",
                "header": "Has scenario analysis impacted your business strategy?",
                "type": "select",
                "options": COMMON_OPTIONS["yes_no"],
                "required": True
            },
            {
                "id": "description",
                "header": "Description of impact on business strategy",
                "type": "textarea",
                "max_length": 2500,
                "condition": {"strategy_impacted": "Yes"}
            }
        ]
    },

    # ------------------------------------------------------------------------
    # 2.2.9 Financial Planning Impact
    # ------------------------------------------------------------------------
    "2.2.9": {
        "title": "Financial planning impact from scenario analysis",
        "response_type": "table",
        "allow_multiple_rows": True,
        "columns": [
            {
                "id": "financial_planning_impacted",
                "header": "Has scenario analysis impacted your financial planning?",
                "type": "select",
                "options": COMMON_OPTIONS["yes_no"],
                "required": True
            },
            {
                "id": "description",
                "header": "Description of impact on financial planning",
                "type": "textarea",
                "max_length": 2500,
                "condition": {"financial_planning_impacted": "Yes"}
            }
        ]
    },

    # ------------------------------------------------------------------------
    # 2.3 Identified Risks
    # ------------------------------------------------------------------------
    "2.3": {
        "title": "Identified environmental risks with substantive financial or strategic impact",
        "response_type": "table",
        "allow_multiple_rows": True,
        "columns": [
            {
                "id": "environmental_issue",
                "header": "Environmental issue",
                "type": "select",
                "options": COMMON_OPTIONS["environmental_issues"],
                "required": True
            },
            {
                "id": "risk_type",
                "header": "Risk type",
                "type": "grouped_select",
                "grouped_options": GROUPED_OPTIONS["risk_types"],
                "required": True
            },
            {
                "id": "primary_risk_driver",
                "header": "Primary risk driver",
                "type": "textarea",
                "max_length": 500
            },
            {
                "id": "description",
                "header": "Description of risk",
                "type": "textarea",
                "max_length": 2500,
                "required": True
            },
            {
                "id": "value_chain_stage",
                "header": "Primary value chain stage affected",
                "type": "select",
                "options": COMMON_OPTIONS["value_chain_stages"]
            },
            {
                "id": "time_horizon",
                "header": "Time horizon",
                "type": "select",
                "options": COMMON_OPTIONS["time_horizons"]
            },
            {
                "id": "likelihood",
                "header": "Likelihood",
                "type": "select",
                "options": [
                    "Virtually certain",
                    "Very likely",
                    "Likely",
                    "More likely than not",
                    "About as likely as not",
                    "Unlikely",
                    "Very unlikely",
                    "Exceptionally unlikely"
                ]
            },
            {
                "id": "magnitude",
                "header": "Magnitude of impact",
                "type": "select",
                "options": ["High", "Medium-high", "Medium", "Medium-low", "Low"]
            },
            {
                "id": "financial_impact",
                "header": "Potential financial impact",
                "type": "number",
                "min_value": 0
            },
            {
                "id": "financial_impact_explanation",
                "header": "Explanation of financial impact",
                "type": "textarea",
                "max_length": 2500
            },
            {
                "id": "management_method",
                "header": "Management method",
                "type": "textarea",
                "max_length": 2500
            },
            {
                "id": "cost_of_management",
                "header": "Cost of management",
                "type": "number",
                "min_value": 0
            }
        ]
    },

    # ------------------------------------------------------------------------
    # 2.4 Identified Opportunities
    # ------------------------------------------------------------------------
    "2.4": {
        "title": "Identified environmental opportunities with substantive financial or strategic impact",
        "response_type": "table",
        "allow_multiple_rows": True,
        "columns": [
            {
                "id": "environmental_issue",
                "header": "Environmental issue",
                "type": "select",
                "options": COMMON_OPTIONS["environmental_issues"],
                "required": True
            },
            {
                "id": "opportunity_type",
                "header": "Opportunity type",
                "type": "grouped_select",
                "grouped_options": GROUPED_OPTIONS["opportunity_types"],
                "required": True
            },
            {
                "id": "primary_opportunity_driver",
                "header": "Primary opportunity driver",
                "type": "textarea",
                "max_length": 500
            },
            {
                "id": "description",
                "header": "Description of opportunity",
                "type": "textarea",
                "max_length": 2500,
                "required": True
            },
            {
                "id": "value_chain_stage",
                "header": "Primary value chain stage affected",
                "type": "select",
                "options": COMMON_OPTIONS["value_chain_stages"]
            },
            {
                "id": "time_horizon",
                "header": "Time horizon",
                "type": "select",
                "options": COMMON_OPTIONS["time_horizons"]
            },
            {
                "id": "likelihood",
                "header": "Likelihood",
                "type": "select",
                "options": [
                    "Virtually certain",
                    "Very likely",
                    "Likely",
                    "More likely than not",
                    "About as likely as not",
                    "Unlikely",
                    "Very unlikely",
                    "Exceptionally unlikely"
                ]
            },
            {
                "id": "magnitude",
                "header": "Magnitude of impact",
                "type": "select",
                "options": ["High", "Medium-high", "Medium", "Medium-low", "Low"]
            },
            {
                "id": "financial_impact",
                "header": "Potential financial impact",
                "type": "number",
                "min_value": 0
            },
            {
                "id": "financial_impact_explanation",
                "header": "Explanation of financial impact",
                "type": "textarea",
                "max_length": 2500
            },
            {
                "id": "strategy_to_realize",
                "header": "Strategy to realize opportunity",
                "type": "textarea",
                "max_length": 2500
            },
            {
                "id": "cost_to_realize",
                "header": "Cost to realize opportunity",
                "type": "number",
                "min_value": 0
            }
        ]
    }
}


def get_schema(question_id: str) -> Optional[Dict[str, Any]]:
    """질문 ID로 스키마 조회"""
    return RESPONSE_SCHEMAS.get(question_id)


def get_all_question_ids() -> List[str]:
    """모든 질문 ID 목록 반환"""
    return list(RESPONSE_SCHEMAS.keys())


def validate_response(question_id: str, response: Dict[str, Any]) -> List[str]:
    """응답 유효성 검증"""
    errors = []
    schema = get_schema(question_id)

    if not schema:
        return [f"Unknown question ID: {question_id}"]

    columns = schema.get("columns", [])

    for col in columns:
        col_id = col["id"]
        col_type = col["type"]
        required = col.get("required", False)

        value = response.get(col_id)

        # Required 체크
        if required and (value is None or value == "" or value == []):
            errors.append(f"Required field '{col_id}' is missing")
            continue

        if value is None or value == "":
            continue

        # 타입별 검증
        if col_type == "select":
            options = col.get("options", [])
            if value not in options:
                errors.append(f"Invalid option '{value}' for '{col_id}'. Valid: {options}")

        elif col_type == "multiselect":
            options = col.get("options", [])
            if isinstance(value, list):
                for v in value:
                    if v not in options:
                        errors.append(f"Invalid option '{v}' for '{col_id}'")
            else:
                errors.append(f"'{col_id}' should be a list")

        elif col_type == "number" or col_type == "percentage":
            min_val = col.get("min_value")
            max_val = col.get("max_value")
            try:
                num_val = float(value)
                if min_val is not None and num_val < min_val:
                    errors.append(f"'{col_id}' must be >= {min_val}")
                if max_val is not None and num_val > max_val:
                    errors.append(f"'{col_id}' must be <= {max_val}")
            except (ValueError, TypeError):
                errors.append(f"'{col_id}' must be a number")

        elif col_type == "textarea":
            max_length = col.get("max_length")
            if max_length and len(str(value)) > max_length:
                errors.append(f"'{col_id}' exceeds max length {max_length}")

    return errors


def export_schema_to_json(output_path: str):
    """스키마를 JSON으로 내보내기"""
    import json

    # Enum을 문자열로 변환
    def convert(obj):
        if isinstance(obj, Enum):
            return obj.value
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(RESPONSE_SCHEMAS, f, ensure_ascii=False, indent=2, default=convert)

    print(f"Schema exported to {output_path}")


if __name__ == "__main__":
    # 테스트: 스키마 정보 출력
    print("CDP Module 2 Response Schemas")
    print("=" * 50)

    for qid, schema in RESPONSE_SCHEMAS.items():
        print(f"\n{qid}: {schema['title']}")
        print(f"  Type: {schema['response_type']}")
        print(f"  Multiple rows: {schema.get('allow_multiple_rows', False)}")
        print(f"  Columns: {len(schema['columns'])}")
        for col in schema['columns'][:3]:  # 처음 3개만
            print(f"    - {col['id']}: {col['type']}")
        if len(schema['columns']) > 3:
            print(f"    ... and {len(schema['columns']) - 3} more")

    # JSON 내보내기
    export_schema_to_json("data/cdp_response_schema.json")
