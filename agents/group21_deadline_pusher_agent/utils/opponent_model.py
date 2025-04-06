from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.DiscreteValueSet import DiscreteValueSet
from geniusweb.issuevalue.Domain import Domain


class OpponentModel:
    def __init__(self, domain: Domain):
        self.offers: list[Bid] = []
        self.domain = domain

        self.issue_estimators = {
            i: IssueEstimator(v) for i, v in domain.getIssuesValues().items()
        }

    def update(self, bid: Bid):
        # keep track of all bids received
        self.offers.append(bid)

        # update all issue estimators with the value that is offered for that issue
        average_value_distances = self.get_average_value_distances()
        for issue_id, issue_estimator in self.issue_estimators.items():
            issue_estimator.update(average_value_distances.get(issue_id, 3))

    def getWeight(self, issue: str) -> float:
        if issue in self.issue_estimators:
            total_issue_weight = 0.0
            for issue_estimator in self.issue_estimators.values():
                total_issue_weight += issue_estimator.weight

            if total_issue_weight == 0.0:
                return 1 / len(self.issue_estimators)

            return self.issue_estimators[issue].weight / total_issue_weight

        return 0.0
    
    def get_average_value_distances(self):
        """Calculate the average distance between the values of the last two bids received.

        Returns:
            dict: A dictionary where the keys are issue ids and the values are the average distances.
        """
        if len(self.offers) < 2:
            return {}

        bid1: Bid = self.offers[-2]
        bid2: Bid = self.offers[-1]

        average_value_distances = {}
        for issue in self.domain.getIssues():
            values: DiscreteValueSet = self.domain.getValues(issue)

            value1 = bid1.getValue(issue)
            value2 = bid2.getValue(issue)

            ind1 = values.getValues().index(value1)
            ind2 = values.getValues().index(value2)

            average_value_distances[issue] = int(min(abs(ind1 - ind2), 4))

        return average_value_distances

class IssueEstimator:
    def __init__(self, value_set: DiscreteValueSet):
        if not isinstance(value_set, DiscreteValueSet):
            raise TypeError(
                "This issue estimator only supports issues with discrete values"
            )

        self.bids_received = 0
        self.weight = 0
        self.avd_to_weight = {
            0: 6,
            1: 4,
            2: 3,
            3: 1,
            4: 0.5,
        }

    def update(self, avd: int):
        self.bids_received += 1
        # update weight based on the last two bids received
        self.weight = self.avd_to_weight[avd]