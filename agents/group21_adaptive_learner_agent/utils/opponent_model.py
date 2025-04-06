import math
from collections import defaultdict

from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.DiscreteValueSet import DiscreteValueSet
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value


class BayesianIssueEstimator:
    """
    Bayesian estimator for a single negotiation issue.
    
    It uses a Dirichlet prior (with all parameters equal to 1) for the possible discrete values.
    The expected posterior probability for each value is calculated and used both as a proxy for
    the opponent's evaluation of that value and to compute an importance weight for the issue.
    """
    def __init__(self, value_set: DiscreteValueSet):
        if not isinstance(value_set, DiscreteValueSet):
            raise TypeError("BayesianIssueEstimator supports only issues with discrete values")
        self.value_set = value_set
        self.n = value_set.size()
        # Initialize counts for each value with 0 (the Dirichlet prior adds one later)
        self.value_counts = {value: 0 for value in value_set.getValues()}
        self.total_count = 0

    def update(self, value: Value):
        """
        Update the count for the given value based on an observed bid.
        """
        self.total_count += 1
        self.value_counts[value] = self.value_counts.get(value, 0) + 1

    def get_value_utility(self, value: Value) -> float:
        """
        Returns the posterior expected probability for the given value.
        This serves as the predicted utility of that value to the opponent.
        """
        # Dirichlet update: (count + 1) / (total_count + n)
        return (self.value_counts.get(value, 0) + 1) / (self.total_count + self.n)

    def get_weight(self) -> float:
        """
        Computes the estimated importance of the issue.
        
        This is done by calculating the (normalized) entropy of the posterior distribution
        over the issue's values. A lower (normalized) entropy indicates a more peaked distribution,
        hence a higher weight (importance) for the issue.
        """
        probabilities = [
            (self.value_counts.get(value, 0) + 1) / (self.total_count + self.n)
            for value in self.value_set.getValues()
        ]
        # Calculate entropy: -sum(p * log(p))
        entropy = -sum(p * math.log(p) for p in probabilities)
        max_entropy = math.log(self.n)  # maximum entropy if the distribution is uniform
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        # Issue weight: lower entropy (more peaked) implies higher importance.
        return 1 - normalized_entropy


class OpponentModel:
    """
    Opponent model using a Bayesian Learning approach.
    
    For each issue, the model maintains a BayesianIssueEstimator to track the opponent's preferences.
    As bids are observed, the model updates these estimators.
    When predicting the utility of a bid for the opponent, the model computes a weighted sum over issues,
    where each issue's contribution is the estimated value utility (from the Bayesian estimator) weighted by
    the issue's importance.
    """
    def __init__(self, domain: Domain):
        self.offers = []
        self.domain = domain
        # Create a BayesianIssueEstimator for each issue in the domain
        self.issue_estimators = {
            issue_id: BayesianIssueEstimator(value_set)
            for issue_id, value_set in domain.getIssuesValues().items()
        }

    def update(self, bid: Bid):
        """
        Update the model with a newly received bid.
        For each issue, the corresponding Bayesian estimator is updated with the offered value.
        """
        self.offers.append(bid)
        for issue_id, estimator in self.issue_estimators.items():
            estimator.update(bid.getValue(issue_id))

    def get_predicted_utility(self, bid: Bid) -> float:
        """
        Predicts the opponent's utility for a given bid.
        
        The prediction is based on the weighted sum of each issue's estimated value utility,
        where the weight for an issue is derived from the peakedness (importance) of its posterior distribution.
        """
        if len(self.offers) == 0 or bid is None:
            return 0.0

        predicted_utility = 0.0
        total_weight = 0.0

        for issue_id, estimator in self.issue_estimators.items():
            # Estimated importance (weight) of the issue
            weight = estimator.get_weight()
            total_weight += weight
            # Predicted utility of the offered value for this issue
            value_util = estimator.get_value_utility(bid.getValue(issue_id))
            predicted_utility += weight * value_util

        # Normalize so that the predicted utility is within [0,1]
        if total_weight > 0:
            predicted_utility /= total_weight
        else:
            predicted_utility = 0.0

        return predicted_utility