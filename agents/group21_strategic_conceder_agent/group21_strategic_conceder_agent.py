import logging
from random import randint
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


class Group21StrategicConcederAgent(DefaultParty):
    """
    Group 21 Strategic Conceder negotiation agent. Uses the following strategies:
    Bidding strategy: Trade-off
    Opponent modelling: KDE
    Acceptance condition: ACnext
    """

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None

        self.reservation_value = 0
        self.max_utility = 1.0
        self.target_utility = self.max_utility
        self.bid_search_attempts = 1000
        self.min_bids_for_opponent_model = 5
        self.last_utility = 1.0
        self.next_bid_difference_margin = 0.05

        self.logger.log(logging.INFO, "party is initialized")


    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()

            self.all_bids_list = AllBidsList(self.domain)
            self.opponent_model = OpponentModel(self.domain)

            # Determine reservation value
            res_bid = self.profile.getReservationBid()
            if res_bid is not None:
                self.reservation_value = float(self.profile.getUtility(res_bid))
            else:
                self.reservation_value = 0

            self.best_bid_overall = self.profile.getReservationBid()
            if self.best_bid_overall is None:
                self.best_bid_overall = self._find_fallback_bid()

            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            self._update_target_utility()
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))


    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )


    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)


    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Group21StrategicConcederAgent: KDE Opponent Model, Trade-off Bidding, AC_next Acceptance"


    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            bid = cast(Offer, action).getBid()

            # update opponent model with bid
            self.opponent_model.update(bid)
            # set bid as last received
            self.last_received_bid = bid


    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        next_bid_to_offer = self.find_bid()
        self.last_utility = self.profile.getUtility(next_bid_to_offer)

        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid, next_bid_to_offer):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            action = Offer(self.me, next_bid_to_offer)

        # send the action
        self.send_action(action)


    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)


    def _update_target_utility(self):
        """ Recalculates the target utility based on time progress using a concession curve."""
        self.target_utility -= 0.0005
        self.target_utility = max(self.reservation_value, self.target_utility)


    def accept_condition(self, received_bid: Bid, next_bid_to_offer) -> bool:
        if received_bid is None:
            return False
        
        received_utility = self.profile.getUtility(received_bid)
        next_offer_utility = self.profile.getUtility(next_bid_to_offer)

        above_reservation = received_utility >= self.reservation_value
        meets_ac_next = received_utility >= next_offer_utility
        return above_reservation and meets_ac_next


    def find_bid(self) -> Bid:
        """
        Finds a bid for the trade-off strategy.
        1. Samples bids.
        2. Filters bids with utility >= current target utility.
        3. Ranks filtered bids by predicted opponent utility (using KDE model).
        4. Returns the best bid found, or falls back to reservation/best possible if needed.
        """
        use_sampling = self.all_bids_list.size() > 5000

        candidate_bids = []
        secondary_candidate_bids = []

        highest_utility_found = self.reservation_value if self.best_bid_overall else 0.0
        if self.best_bid_overall:
            highest_utility_found = float(self.profile.getUtility(self.best_bid_overall))

        # Try to find bids meeting the target utility
        attempts = self.bid_search_attempts if use_sampling else self.all_bids_list.size()
        for i in range(attempts):
            if use_sampling:
                bid = self.all_bids_list.get(randint(0, self.all_bids_list.size() - 1))
            else:
                bid = self.all_bids_list.get(i)

            my_utility = float(self.profile.getUtility(bid))

            # Keep track of the best bid found so far (in case no bids meet the target)
            if my_utility > highest_utility_found:
                highest_utility_found = my_utility
                self.best_bid_overall = bid

            # Check if the bid meets the target utility criteria
            if my_utility >= self.target_utility and abs(my_utility - self.target_utility) <= self.next_bid_difference_margin:
                if my_utility >= self.last_utility:
                    candidate_bids.append(bid)
                secondary_candidate_bids.append(bid)
            # Optimization: If sampling and found enough candidates, stop early
            if use_sampling and len(candidate_bids) > 100:
                break

        # If suitable candidates found that increase our utility, rank them by predicted opponent utility
        if candidate_bids:
            best_tradeoff_bid = None
            max_opponent_utility = -1.0

            for bid in candidate_bids:
                if self.opponent_model:
                    opponent_utility = self.opponent_model.get_predicted_utility(bid)
                else:
                    opponent_utility = 0.0 # Cannot predict if model not ready

                if opponent_utility > max_opponent_utility:
                    max_opponent_utility = opponent_utility
                    best_tradeoff_bid = bid

            if best_tradeoff_bid is not None:
                return best_tradeoff_bid
            else:
                # Should not happen if candidate_bids is not empty, but fallback
                return candidate_bids[0]
            
        # If other suitable candidates (decreasing our utility) are found, look for the optimal opponent utility prediction among them
        if secondary_candidate_bids:
            best_tradeoff_bid = None
            max_opponent_utility = -1.0

            for bid in secondary_candidate_bids:
                if self.opponent_model:
                    opponent_utility = self.opponent_model.get_predicted_utility(bid)
                else:
                    opponent_utility = 0.0 # Cannot predict if model not ready

                if opponent_utility > max_opponent_utility:
                    max_opponent_utility = opponent_utility
                    best_tradeoff_bid = bid

            if best_tradeoff_bid is not None:
                return best_tradeoff_bid
            else:
                # Should not happen if candidate_bids is not empty, but fallback
                return secondary_candidate_bids[0]

        else:
            # No bids met the target utility criteria, return the best bid found overall
            # Ensure the best bid is at least the reservation bid
            if self.best_bid_overall is None: # Very unlikely case
                self.logger.error("Could not find any valid bid, not even reservation bid!")
                # Attempt to generate *any* bid as last resort
                return self.all_bids_list.get(0)

            my_utility_best_overall = float(self.profile.getUtility(self.best_bid_overall))
            if my_utility_best_overall < self.reservation_value:
                res_bid = self.profile.getReservationBid()
                if res_bid:
                    # self.logger.warning(f"Best overall bid U={my_utility_best_overall:.3f} below reservation. Returning reservation bid.")
                    return res_bid
                else:
                    # self.logger.warning(f"Best overall bid U={my_utility_best_overall:.3f} below target, and no reservation bid. Returning best overall.")
                    return self.best_bid_overall # Return best found even if below reservation if no res bid exists

            # self.logger.log(logging.DEBUG, f"No bids met target utility {self.target_utility:.3f}. Returning best found bid with utility {my_utility_best_overall:.3f}")
            return self.best_bid_overall


    def _find_fallback_bid(self) -> Bid:
        """ Finds a bid with the lowest possible positive utility as a fallback """
        if self.all_bids_list is None:
            self.all_bids_list = AllBidsList(self.domain)

        min_util_bid = self.all_bids_list.get(0)
        min_util = float(self.profile.getUtility(min_util_bid))

        for i in range(1, self.all_bids_list.size()):
            bid = self.all_bids_list.get(i)
            util = float(self.profile.getUtility(bid))
            if util < min_util :
                min_util = util
                min_util_bid = bid
        return min_util_bid
