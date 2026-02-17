#!/usr/bin/env python3
"""
Age Correction Module
=====================

Applies regression-based corrections to age estimates from vision models.

The ViT age estimator shows systematic bias for female subjects:
- Underestimates young ages (detects lower than true)
- Overestimates older teen ages (detects higher than true)
- Roughly accurate around age 12

The correction is based on empirical regression analysis.

Usage:
    from age_correction import correct_age, AgeCorrector
    
    # Simple usage
    true_age = correct_age(detected_age=19, gender='female')
    
    # With custom parameters
    corrector = AgeCorrector(coefficients={'a': 0.005, 'b': 0.02, 'c': 0.1})
    true_age = corrector.correct(detected_age=19, gender='female')

Author: Pipeline Tools
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CorrectionResult:
    """Result of age correction."""
    detected_age: float
    corrected_age: float
    skew: float
    gender: str
    correction_applied: bool
    confidence_note: str


# =============================================================================
# DEFAULT COEFFICIENTS
# =============================================================================

# Cubic regression coefficients for female subjects
# Model: skew = a*(x-12)³ + b*(x-12)² + c*(x-12)
# where x = detected_age, skew = detected - true
# Fitted to empirical data points:
#   - At x=1, skew=-7 (true=8)
#   - At x=12, skew=0 (true=12)  
#   - At x=19, skew=5 (true=14)

DEFAULT_FEMALE_COEFFICIENTS = {
    'center': 12.0,       # Age where skew = 0
    'a': 0.00885458,      # Cubic term (controls curvature at extremes)
    'b': 0.03974734,      # Quadratic term (asymmetry)
    'c': 0.00217979,      # Linear term (overall slope)
}

# Age range where correction is reliable
# Beyond ~20, the cubic extrapolation becomes unreliable
RELIABLE_RANGE = (1, 20)  # Outside this range, correction is uncertain

# Range with irregular expression (higher uncertainty)
IRREGULAR_RANGE = (8, 15)

# Maximum correction magnitude (prevents extreme extrapolation)
MAX_CORRECTION = 10.0  # Don't correct more than 10 years in either direction


# =============================================================================
# CORRECTION FUNCTIONS
# =============================================================================

def compute_skew(detected_age: float, 
                 coefficients: Optional[Dict] = None) -> float:
    """
    Compute the expected skew (detected - true) for a given detected age.
    
    Uses cubic polynomial centered at the crossover point.
    
    Args:
        detected_age: Age as reported by the model
        coefficients: Dict with 'center', 'a', 'b', 'c' keys
        
    Returns:
        Expected skew value (positive = model reads high, negative = model reads low)
    """
    if coefficients is None:
        coefficients = DEFAULT_FEMALE_COEFFICIENTS
    
    center = coefficients.get('center', 12.0)
    a = coefficients.get('a', 0.0045)
    b = coefficients.get('b', 0.015)
    c = coefficients.get('c', 0.08)
    
    # Centered variable
    x = detected_age - center
    
    # Cubic polynomial
    skew = a * (x ** 3) + b * (x ** 2) + c * x
    
    return skew


def correct_age(detected_age: float,
                gender: Optional[str] = None,
                coefficients: Optional[Dict] = None,
                apply_to_males: bool = False) -> CorrectionResult:
    """
    Apply age correction based on regression model.
    
    Args:
        detected_age: Age as reported by the vision model
        gender: 'female', 'male', or None
        coefficients: Custom regression coefficients (uses defaults if None)
        apply_to_males: If True, also applies correction to males (default False)
        
    Returns:
        CorrectionResult with corrected age and metadata
    """
    gender_lower = (gender or '').lower()
    is_female = gender_lower in ('female', 'woman', 'f')
    is_male = gender_lower in ('male', 'man', 'm')
    
    # Determine if correction should be applied
    apply_correction = is_female or (apply_to_males and is_male)
    
    if not apply_correction:
        return CorrectionResult(
            detected_age=detected_age,
            corrected_age=detected_age,
            skew=0.0,
            gender=gender or 'unknown',
            correction_applied=False,
            confidence_note="No correction applied (male or unknown gender)"
        )
    
    # Compute skew
    skew = compute_skew(detected_age, coefficients)
    
    # Clamp skew to prevent extreme corrections outside reliable range
    skew = max(-MAX_CORRECTION, min(MAX_CORRECTION, skew))
    
    # Apply correction: true = detected - skew
    corrected = detected_age - skew
    
    # Clamp to reasonable age range
    corrected = max(1.0, min(100.0, corrected))
    
    # Determine confidence note
    if detected_age < RELIABLE_RANGE[0]:
        note = f"Low confidence: detected age {detected_age:.1f} below reliable range"
    elif detected_age > RELIABLE_RANGE[1]:
        note = f"Low confidence: detected age {detected_age:.1f} above reliable range"
    elif IRREGULAR_RANGE[0] <= detected_age <= IRREGULAR_RANGE[1]:
        note = f"Moderate confidence: detected age {detected_age:.1f} in irregular range"
    else:
        note = "Normal confidence"
    
    return CorrectionResult(
        detected_age=detected_age,
        corrected_age=round(corrected, 1),
        skew=round(skew, 2),
        gender=gender or 'unknown',
        correction_applied=True,
        confidence_note=note
    )


# =============================================================================
# CORRECTOR CLASS
# =============================================================================

class AgeCorrector:
    """
    Configurable age corrector with custom coefficients.
    
    Example:
        corrector = AgeCorrector()
        result = corrector.correct(detected_age=19, gender='female')
        print(f"True age estimate: {result.corrected_age}")
    """
    
    def __init__(self, 
                 coefficients: Optional[Dict] = None,
                 apply_to_males: bool = False):
        """
        Initialize corrector.
        
        Args:
            coefficients: Custom regression coefficients
            apply_to_males: Whether to apply correction to male subjects
        """
        self.coefficients = coefficients or DEFAULT_FEMALE_COEFFICIENTS.copy()
        self.apply_to_males = apply_to_males
    
    def correct(self, detected_age: float, gender: Optional[str] = None) -> CorrectionResult:
        """Apply correction to a single age estimate."""
        return correct_age(
            detected_age=detected_age,
            gender=gender,
            coefficients=self.coefficients,
            apply_to_males=self.apply_to_males
        )
    
    def correct_batch(self, 
                      ages: list,
                      genders: Optional[list] = None) -> list:
        """
        Apply correction to multiple age estimates.
        
        Args:
            ages: List of detected ages
            genders: List of genders (same length as ages, or None)
            
        Returns:
            List of CorrectionResult objects
        """
        if genders is None:
            genders = [None] * len(ages)
        
        return [self.correct(age, gender) for age, gender in zip(ages, genders)]
    
    def get_correction_curve(self, 
                             age_range: Tuple[float, float] = (1, 40),
                             step: float = 1.0) -> Dict:
        """
        Generate the correction curve for visualization.
        
        Returns dict with:
            - detected_ages: list of x values
            - skews: list of skew values
            - corrected_ages: list of corrected ages
        """
        detected_ages = np.arange(age_range[0], age_range[1] + step, step)
        skews = [compute_skew(x, self.coefficients) for x in detected_ages]
        corrected = [x - s for x, s in zip(detected_ages, skews)]
        
        return {
            'detected_ages': detected_ages.tolist(),
            'skews': skews,
            'corrected_ages': corrected,
            'coefficients': self.coefficients,
        }
    
    def fit_from_points(self, 
                        points: list,
                        center: float = 12.0) -> Dict:
        """
        Fit new coefficients from known data points.
        
        Args:
            points: List of (detected_age, true_age) tuples
            center: Age where skew should be zero
            
        Returns:
            New coefficients dict
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit cubic")
        
        # Convert to skew form
        # skew = detected - true
        x_data = []
        y_data = []
        for detected, true in points:
            x_data.append(detected - center)
            y_data.append(detected - true)
        
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        
        # Fit polynomial: y = ax³ + bx² + cx (no constant term, passes through origin)
        # Use least squares
        X = np.column_stack([x_data**3, x_data**2, x_data])
        coeffs, _, _, _ = np.linalg.lstsq(X, y_data, rcond=None)
        
        new_coefficients = {
            'center': center,
            'a': float(coeffs[0]),
            'b': float(coeffs[1]),
            'c': float(coeffs[2]),
        }
        
        self.coefficients = new_coefficients
        return new_coefficients


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_age_category(age: float) -> str:
    """Convert numeric age to category."""
    if age is None:
        return "unknown"
    
    categories = {
        'infant': (0, 2),
        'toddler': (2, 4),
        'child': (4, 9),
        'adolescent': (9, 13),
        'early_teen': (13, 16),
        'late_teen': (16, 19),
        'young_adult': (19, 30),
        'adult': (30, 45),
        'middle_aged': (45, 60),
        'senior': (60, 120),
    }
    
    for category, (min_age, max_age) in categories.items():
        if min_age <= age < max_age:
            return category
    
    return "senior" if age >= 60 else "unknown"


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("Age Correction Module - Test")
    print("=" * 50)
    
    corrector = AgeCorrector()
    
    # Test the specified data points
    test_cases = [
        (1, 'female'),   # Should correct to ~8
        (12, 'female'),  # Should stay ~12
        (19, 'female'),  # Should correct to ~14
        (25, 'female'),  # Young adult
        (30, 'female'),  # Adult
        (15, 'male'),    # Male - no correction
    ]
    
    print(f"\n{'Detected':>10} {'Gender':>8} {'Corrected':>10} {'Skew':>8} {'Note'}")
    print("-" * 70)
    
    for detected, gender in test_cases:
        result = corrector.correct(detected, gender)
        print(f"{result.detected_age:>10.1f} {result.gender:>8} "
              f"{result.corrected_age:>10.1f} {result.skew:>8.2f}  {result.confidence_note}")
    
    # Print correction curve
    print("\n\nCorrection Curve (females):")
    print("-" * 50)
    curve = corrector.get_correction_curve((1, 30), step=2)
    print(f"{'Detected':>10} {'Skew':>10} {'Corrected':>10}")
    for d, s, c in zip(curve['detected_ages'], curve['skews'], curve['corrected_ages']):
        print(f"{d:>10.1f} {s:>10.2f} {c:>10.1f}")
