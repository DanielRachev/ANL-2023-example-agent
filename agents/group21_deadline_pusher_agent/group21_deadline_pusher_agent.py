import logging
from time import time
from typing import cast

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
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
from geniusweb.profile.utilityspace.NumberValueSetUtilities import NumberValueSetUtilities
from geniusweb.profile.utilityspace.DiscreteValueSetUtilities import DiscreteValueSetUtilities
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel


class DeadlinePusher(DefaultParty):
    """
    Implementation of the deadline pusher agent.
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
        self.logger.log(logging.INFO, "party is initialized")

        # Time-dependent bidding strategy
        self.time_cutoff = 0.95
        self.kappa = 0.1
        self.beta = 0.2 # Boulware tactics family

        # Guessing heuristic opponent modelling
        self.negotiation_speed = 0.05
        self.minimal_utility = 0.8
        self.tau_gen = 0.1

    def notifyChange(self, data: Inform):
        """
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            data (Inform): Contains either a request for action or information.
        """

        # Settings message is the first message that will be send to your
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
            reservation_bid = self.profile.getReservationBid()
            if reservation_bid:
                self.minimal_utility = float(self.profile.getUtility(reservation_bid)) # Get utility of reservation bid
            self.domain = self.profile.getDomain()

            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

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
        """
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

    def getDescription(self) -> str:
        """
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Deadline pusher agent"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()

            # update opponent model with bid
            self.opponent_model.update(bid)
            # set bid as last received
            self.last_received_bid = bid

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self.find_bid()
            action = Offer(self.me, bid)

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

    ###########################################################################################
    ################################### Agent methods below ###################################
    ###########################################################################################

    def get_time(self) -> float:
        """Utility function to get the time progress of the negotiation."""
        return self.progress.get(time() * 1000)

    def accept_condition(self, bid: Bid) -> bool:
        """Time dependent acceptance condition."""
        if bid is None:
            return False
        
        return self.get_time() > self.time_cutoff

    def compute_alpha(self, t):
        """Implementation of a polynomial alpha function."""
        return self.kappa + (1 - self.kappa) * (t ** (1 / self.beta))
        # Exponential implementation
        # return exp(log(self.kappa) * (1 - t) ** self.beta)
    
    def get_value_from_utility(self, utility, target_utility):
        """Get the value from the utility function given a target utility."""
        if isinstance(utility, NumberValueSetUtilities):
            low_val = utility.getLowValue()
            high_val = utility.getHighValue()
            low_util = utility.getLowUtility()
            high_util = utility.getHighUtility()

            if high_val == low_val or high_util == low_util:
                return low_val

            # Linear interpolation formula: y = y0 + ((y1 - y0) * (x - x0)) / (x1 - x0)
            # We want to solve for x: value, given y = target_utility
            value = low_val + ((high_val - low_val) * (target_utility - low_util)) / (high_util - low_util)
            return value

        if isinstance(utility, DiscreteValueSetUtilities):
            utilities = utility.getUtilities()
            # Find the key (value) with the closest utility
            closest_val = min(utilities.items(), key=lambda item: abs(float(item[1]) - target_utility))[0]
            return closest_val

        return None
    
    def find_bid(self) -> Bid:
        """Find a bid to propose as counter offer."""
        t = self.get_time()
        time_bid_dict = {}

        utilities = self.profile.getUtilities()

        for issue, utility in utilities.items():
            if isinstance(utility, NumberValueSetUtilities):
                utility = cast(NumberValueSetUtilities, utility)

                min_value = utility.getLowValue()
                max_value = utility.getHighValue()
                
                if utility.getLowUtility() < utility.getHighUtility():
                    # Monotonically increasing utility function
                    time_bid_dict[issue] = min_value + (1 - self.compute_alpha(t)) * (max_value - min_value)
                else:
                    # Monotonically decreasing utility function
                    time_bid_dict[issue] = min_value + self.compute_alpha(t) * (max_value - min_value)
            elif isinstance(utility, DiscreteValueSetUtilities):
                utility = cast(DiscreteValueSetUtilities, utility)

                values_with_utils = list(utility.getUtilities().items())
                values_with_utils.sort(key=lambda x: x[1], reverse=True)

                ind = int(self.compute_alpha(t) * (len(values_with_utils) - 1))

                time_bid_dict[issue] = values_with_utils[ind][0]
                
        time_bid = Bid(time_bid_dict)

        if not self.last_received_bid:
            return time_bid

        utility_bs = float(self.profile.getUtility(time_bid))
        utility_bo = float(self.profile.getUtility(self.last_received_bid))

        concession_step = self.negotiation_speed * (1.0 - self.minimal_utility / (utility_bs + 1e-6)) * (utility_bo - utility_bs + 1e-6)
        
        target_utility = utility_bs + concession_step

        normalization_factor = 1e-6
        for issue, utility in utilities.items():
            weight_s = float(self.profile.getWeight(issue))
            evaluation_bs_j = float(utility.getUtility(time_bid.getValue(issue)))

            alpha_j = (1.0 - weight_s) * (1.0 - evaluation_bs_j)

            normalization_factor += weight_s * alpha_j

        final_bid_dict = {}
        for issue, utility in utilities.items():
            weight_s = float(self.profile.getWeight(issue))
            weight_o = self.opponent_model.getWeight(issue)

            delta_j = (weight_o - weight_s) / (weight_o + weight_s)
            tau_j = self.tau_gen * (1.0 + delta_j)
        
            evaluation_bs_j = float(utility.getUtility(time_bid.getValue(issue)))
            evaluation_bo_j = float(utility.getUtility(self.last_received_bid.getValue(issue)))

            alpha_j = (1.0 - weight_s) * (1.0 - evaluation_bs_j)
            basic_target_evaluation_j = evaluation_bs_j + (alpha_j / normalization_factor) * (target_utility - utility_bs)
            target_evaluation_j = (1.0 - tau_j) * basic_target_evaluation_j + tau_j * evaluation_bo_j

            final_bid_dict[issue] = self.get_value_from_utility(utility, target_evaluation_j)

        final_bid = Bid(final_bid_dict)

        return final_bid