"""
EdgePPE-Vision: Compliance Logic Engine
Implements the safety compliance decision-making rules as defined in Proposal Section 9.3.
"""

from typing import List, Dict, Any, Tuple
from .config import MANDATORY_GEAR, VIOLATION_CLASSES


class ComplianceEngine:
    """
    Evaluates detection results to determine overall safety gear compliance.
    """
    def __init__(
        self,
        mandatory_gear: List[str] = None,
        violation_classes: List[str] = None,
        aliases: Dict[str, str] = None,
        enforce_mask: bool = False
    ):
        self.mandatory_gear = mandatory_gear or MANDATORY_GEAR
        self.violation_classes = set(violation_classes or VIOLATION_CLASSES)
        self.enforce_mask = enforce_mask
        
        # Standard alias map for dataset interoperability
        self.aliases = aliases or {
            "helmet": "Hard Hat",
            "hardhat": "Hard Hat",
            "hard-hat": "Hard Hat",
            "no-helmet": "NO-Hard Hat",
            "no-hardhat": "NO-Hard Hat",
            "vest": "Safety Vest",
            "safety-vest": "Safety Vest",
            "safety vest": "Safety Vest",
            "no-safety-vest": "NO-Safety Vest",
            "no-vest": "NO-Safety Vest",
            "facemask": "Mask",
            "face-mask": "Mask",
            "no-facemask": "NO-Mask",
            "no_mask": "NO-Mask"
        }

    def normalize_name(self, name: str) -> str:
        """Map alternative dataset labels to standard classes"""
        key = name.strip().lower()
        return self.aliases.get(key, name)

    def evaluate(self, detections: List[Dict[str, Any]], enforce_mask: bool = False) -> Dict[str, Any]:
        """
        Evaluate list of detections against workplace safety rules.
        
        Rules:
        1. If ANY active violation tag is detected (NO-Hard Hat, NO-Safety Vest, or NO-Mask if enforce_mask=True),
           overall status is NON_COMPLIANT.
        2. If mandatory gear (Hard Hat, Safety Vest) is missing when a Person or gear is detected, status is NON_COMPLIANT.
        3. If mandatory gear is present and NO violation tags detected, status is COMPLIANT.
        """
        detected_classes = set()
        violations_found = []
        mandatory_present = []
        
        for det in detections:
            raw_name = det.get("class_name", "")
            norm_name = self.normalize_name(raw_name)
            det["class_name"] = norm_name  # Update detection with normalized name
            detected_classes.add(norm_name)
            
            # Check for violation tags
            if norm_name in self.violation_classes or norm_name.upper().startswith("NO-"):
                if norm_name == "NO-Mask" and not enforce_mask:
                    # Face masks optional unless explicit policy active
                    continue
                violations_found.append(norm_name)
                
        # Check mandatory gear
        for gear in self.mandatory_gear:
            norm_gear = self.normalize_name(gear)
            if norm_gear in detected_classes:
                mandatory_present.append(norm_gear)
                
        missing_mandatory = [
            gear for gear in self.mandatory_gear 
            if self.normalize_name(gear) not in detected_classes
        ]

        # Determine overall compliance status
        if violations_found:
            is_compliant = False
            status = "NON_COMPLIANT"
            summary = f"Safety Violation: Detected {', '.join(sorted(set(violations_found)))}."
        elif missing_mandatory and (len(detections) > 0 or "Person" in detected_classes):
            is_compliant = False
            status = "NON_COMPLIANT"
            summary = f"Missing Mandatory PPE: {', '.join(missing_mandatory)}."
        elif not detections:
            is_compliant = True
            status = "COMPLIANT"
            summary = "No safety violations or personnel detected in frame."
        else:
            is_compliant = True
            status = "COMPLIANT"
            summary = f"Compliant: All required PPE verified ({', '.join(mandatory_present)})."

        return {
            "status": status,
            "is_compliant": is_compliant,
            "mandatory_present": sorted(list(set(mandatory_present))),
            "missing_mandatory": missing_mandatory,
            "violations_detected": sorted(list(set(violations_found))),
            "summary": summary
        }
