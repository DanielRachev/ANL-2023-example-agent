import logging
from random import randint
from time import time
from typing import cast

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel


class Group21AdaptiveLearnerAgent(DefaultParty):
    """
    Group21Agent implements an almost fully behavior-dependent negotiation strategy.
    In addition to adapting over time, this agent:
      - Tracks the opponent's initial bid utility (via the opponent model) to assess concession.
      - Adjusts its acceptance threshold based on how quickly the opponent is conceding.
      - Weights the opponent's predicted utility in bid scoring as a function of observed concession.
    """

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        # Core negotiation elements (set during negotiation setup)
        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        # Tracking opponent actions
        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.initial_opponent_utility: float = None

        # Adaptive strategy parameters
        self.initial_alpha: float = 1.0   # weight on our utility early in negotiation
        self.final_alpha: float = 0.5     # weight on our utility near deadline
        self.eps: float = 0.1             # time pressure factor

        # Reservation value for our own utility: we will only consider bids with at least this utility.
        self.reservation_value: float = 0.65

    def notifyChange(self, data: Inform):
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()
            self.progress = self.settings.getProgress()
            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            profile_connection.close()

        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            if action.getActor() != self.me:
                self.other = str(action.getActor()).rsplit("_", 1)[0]
                self.opponent_action(action)

        elif isinstance(data, YourTurn):
            self.my_turn()

        elif isinstance(data, Finished):
            self.save_data()
            self.logger.log(logging.INFO, "Negotiation finished, terminating agent.")
            super().terminate()

    def getCapabilities(self) -> Capabilities:
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"])
        )

    def getDescription(self) -> str:
        return ("Group21 Behavior-Dependent Adaptive Learner Negotiation Agent that adjusts "
                "its thresholds and bid scoring based on observed opponent concession behavior.")

    def send_action(self, action: Action):
        self.getConnection().send(action)

    def opponent_action(self, action: Action):
        if isinstance(action, Offer):
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)
            bid = cast(Offer, action).getBid()
            self.opponent_model.update(bid)
            if self.initial_opponent_utility is None:
                self.initial_opponent_utility = self.opponent_model.get_predicted_utility(bid)
                self.logger.log(logging.DEBUG, f"Initial opponent utility: {self.initial_opponent_utility:.3f}")
            self.last_received_bid = bid

    def my_turn(self):
        if self.accept_condition(self.last_received_bid):
            action = Accept(self.me, self.last_received_bid)
        else:
            bid = self.find_bid()
            action = Offer(self.me, bid)
        self.send_action(action)

    def save_data(self):
        data = "Behavior-dependent adaptive learning data stored for analysis."
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    def accept_condition(self, bid: Bid) -> bool:
        if bid is None:
            return False
        progress = self.progress.get(time() * 1000)
        threshold = self.compute_acceptance_threshold(progress)
        current_util = self.profile.getUtility(bid)
        self.logger.log(logging.DEBUG, f"Acceptance threshold: {threshold:.3f}, bid util: {current_util:.3f}")
        return current_util >= threshold

    def compute_acceptance_threshold(self, progress: float) -> float:
        initial_threshold = 0.95
        final_threshold = 0.70
        base_threshold = initial_threshold - (initial_threshold - final_threshold) * progress
        if self.initial_opponent_utility is not None and self.last_received_bid is not None:
            current_opp_util = self.opponent_model.get_predicted_utility(self.last_received_bid)
            concession = (self.initial_opponent_utility - current_opp_util) / self.initial_opponent_utility
            concession = max(0.0, min(concession, 1.0))
            behavior_adjustment = concession * 0.10
            base_threshold -= behavior_adjustment
            self.logger.log(logging.DEBUG, f"Opponent concession: {concession:.3f}, behavior adjustment: {behavior_adjustment:.3f}")
        # Ensure threshold does not fall below our reservation value
        return max(base_threshold, self.reservation_value)

    def find_bid(self) -> Bid:
        all_bids = AllBidsList(self.profile.getDomain())
        best_bid_score = -1.0
        best_bid = None
        attempts = 500
        # First pass: consider only bids with our utility above the reservation value.
        for _ in range(attempts):
            bid = all_bids.get(randint(0, all_bids.size() - 1))
            our_util = self.profile.getUtility(bid)
            if our_util < self.reservation_value:
                continue
            bid_score = self.score_bid(bid)
            if bid_score > best_bid_score:
                best_bid_score = bid_score
                best_bid = bid
        # Fallback: if no bid meets the reservation value, pick the highest scoring bid regardless.
        if best_bid is None:
            for _ in range(attempts):
                bid = all_bids.get(randint(0, all_bids.size() - 1))
                bid_score = self.score_bid(bid)
                if bid_score > best_bid_score:
                    best_bid_score = bid_score
                    best_bid = bid
        return best_bid

    def score_bid(self, bid: Bid) -> float:
        progress = self.progress.get(time() * 1000)
        our_util = float(self.profile.getUtility(bid))
        alpha = self.initial_alpha - (self.initial_alpha - self.final_alpha) * progress
        time_pressure = 1.0 - progress ** (1 / (self.eps + 1e-6))
        score = alpha * time_pressure * our_util
        if self.opponent_model is not None:
            opponent_util = self.opponent_model.get_predicted_utility(bid)
            if self.initial_opponent_utility is not None:
                concession = (self.initial_opponent_utility - opponent_util) / self.initial_opponent_utility
                concession = max(0.0, min(concession, 1.0))
                behavior_weight = 0.5 + concession * 0.5
            else:
                behavior_weight = 0.5
            score += (1.0 - alpha * time_pressure) * behavior_weight * opponent_util
        return score
