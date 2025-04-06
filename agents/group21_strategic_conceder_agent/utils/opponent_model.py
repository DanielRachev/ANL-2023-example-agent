import numpy as np
from scipy.stats import gaussian_kde
from collections import defaultdict

from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.DiscreteValueSet import DiscreteValueSet
from geniusweb.issuevalue.Domain import Domain
from geniusweb.utils import toStr


MIN_SAMPLES_FOR_KDE = 5
EPSILON = 1e-6


class OpponentModel:
    """
    Opponent model using Kernel Density Estimation (KDE).
    """
    def __init__(self, domain: Domain):
        self.offers = []
        self.domain = domain
        self.issue_info = {}

        for issue_name, value_set in domain.getIssuesValues().items():
            if not isinstance(value_set, DiscreteValueSet):
                # This model only supports discrete issues
                continue

            # Create a map from a discrete value to an integer value (index) and vice-versa
            values = list(value_set.getValues())
            value_map = {value: i for i, value in enumerate(values)}
            inverse_value_map = {i: value for value, i in value_map.items()}

            self.issue_info[issue_name] = {
                "values": values,
                "value_map": value_map,
                "inverse_value_map": inverse_value_map,
                "offered_values_numeric": [],
                "weight": 1.0 / len(domain.getIssues()),
                "value_utilities": defaultdict(lambda: 0.0),
                "kde": None,
                "num_possible_values": len(values),
            }


    def update(self, bid: Bid):
        """
        Update the model with a new bid received.
        """
        self.offers.append(bid)

        # Update counts and offered values for each issue
        for issue_name, info in self.issue_info.items():
            value = bid.getValue(issue_name)
            if value is not None and value in info["value_map"]:
                numeric_value = info["value_map"][value]
                info["offered_values_numeric"].append(numeric_value)

        # After enough bids, recalculate weights and utilities
        if len(self.offers) >= MIN_SAMPLES_FOR_KDE:
            self._recalculate_weights_and_utilities()


    def _recalculate_weights_and_utilities(self):
        """
        Recalculate issue weights based on variance and value utilities using KDE.
        """
        # === Estimate issue weights based on variance ===
        valid_inverse_variances = {}
        issues_with_insufficient_data = []

        for issue_name, info in self.issue_info.items():
            offered_values_numeric = info["offered_values_numeric"]
            if len(offered_values_numeric) < 2:
                issues_with_insufficient_data.append(issue_name)
                continue

            variance = np.var(offered_values_numeric)

            inverse_variance = 1.0 / (variance + EPSILON)
            valid_inverse_variances[issue_name] = inverse_variance

        total_inverse_variance = sum(valid_inverse_variances.values())
        total_weight_without_unchanged = 1 - sum(self.issue_info[issue_name]["weight"] for issue_name in issues_with_insufficient_data)

        # Normalize weights to sum to 1
        if total_inverse_variance > EPSILON:
            for issue_name, inverse_variance in valid_inverse_variances.items():
                self.issue_info[issue_name]["weight"] = (inverse_variance / total_inverse_variance) * total_weight_without_unchanged
        else:
            num_issues = len(self.issue_info)
            uniform_weight = 1.0 / num_issues if num_issues > 0 else 1.0
            for issue_name in self.issue_info.keys():
                self.issue_info[issue_name]["weight"] = uniform_weight


        # === Estimate Value Utilities using KDE ===
        for issue_name, info in self.issue_info.items():
            offered_values_numeric = info["offered_values_numeric"]

            if len(offered_values_numeric) < 2:
                # Assign zero utility if insufficient data
                for value in info["values"]:
                    info["value_utilities"][value] = 0.0
                continue

            try:
                # Use gaussian_kde for density estimation
                # Add tiny noise if all points are identical, KDE fails otherwise
                if np.all(np.array(offered_values_numeric) == offered_values_numeric[0]):
                   kde_data = np.array(offered_values_numeric) + np.random.normal(0, 0.01, size=len(offered_values_numeric))
                else:
                   kde_data = offered_values_numeric

                # Fit KDE - bandwidth selection 'scott' or 'silverman' are common defaults
                kde = gaussian_kde(kde_data, bw_method='scott')
                info["kde"] = kde

                # Evaluate KDE at each possible value index
                value_indices = list(info["inverse_value_map"].keys())
                densities = kde.evaluate(value_indices)

                # Normalize densities to get utilities (max density -> utility 1)
                max_density = np.max(densities)
                if max_density > EPSILON:
                    for i, idx in enumerate(value_indices):
                        value = info["inverse_value_map"][idx]
                        info["value_utilities"][value] = densities[i] / max_density
                else: # If max density is zero, assign zero utility
                    for value in info["values"]:
                        info["value_utilities"][value] = 0.0

            except Exception as e:
                self.logger.error(f"KDE calculation failed for issue {issue_name}: {e}")
                # Fallback: assign zero utility
                for value in info["values"]:
                    info["value_utilities"][value] = 0.0


    def get_predicted_utility(self, bid: Bid) -> float:
        """
        Predict the opponent's utility for a given bid based on the current model.
        """
        if bid is None or not self.offers:
            return 0.0 # Cannot predict utility without a bid or data

        predicted_utility = 0.0
        try:
            for issue_name, info in self.issue_info.items():
                value = bid.getValue(issue_name)
                if value is not None:
                    # Get estimated weight for the issue
                    weight = info["weight"]
                    # Get estimated utility for the value within the issue
                    value_utility = info["value_utilities"].get(value, 0.0) # Default to 0 if value somehow not in map
                    predicted_utility += weight * value_utility

            # Clamp utility between 0 and 1
            predicted_utility = max(0.0, min(1.0, predicted_utility))
            return predicted_utility

        except Exception as e:
             # Catch potential errors during calculation
            self.logger.error(f"Error predicting utility for bid {toStr(bid)}: {e}")
            return 0.0 # Return safe value on error
