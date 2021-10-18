# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default data for dynamic routing game."""

from open_spiel.python.games import dynamic_routing_utils

LINE_NETWORK = dynamic_routing_utils.Network({
    "bef_O": "O",
    "O": ["A"],
    "A": ["D"],
    "D": ["aft_D"],
    "aft_D": []
})

LINE_NETWORK_VEHICLES_DEMAND = [
    dynamic_routing_utils.Vehicle("bef_O->O", "D->aft_D") for _ in range(2)
]

LINE_NETWORK_OD_DEMAND = [
    dynamic_routing_utils.OriginDestinationDemand("bef_O->O", "D->aft_D", 0,
                                                  100)
]

BRAESS_NUM_PLAYER = 5
BRAESS_NETWORK = dynamic_routing_utils.Network(
    {
        "O": "A",
        "A": ["B", "C"],
        "B": ["C", "D"],
        "C": ["D"],
        "D": ["E"],
        "E": []
    },
    node_position={
        "O": (0, 0),
        "A": (1, 0),
        "B": (2, 1),
        "C": (2, -1),
        "D": (3, 0),
        "E": (4, 0)
    },
    bpr_a_coefficient={
        "O->A": 0,
        "A->B": 1.0,
        "A->C": 0,
        "B->C": 0,
        "B->D": 0,
        "C->D": 1.0,
        "D->E": 0
    },
    bpr_b_coefficient={
        "O->A": 1.0,
        "A->B": 1.0,
        "A->C": 1.0,
        "B->C": 1.0,
        "B->D": 1.0,
        "C->D": 1.0,
        "D->E": 1.0
    },
    capacity={
        "O->A": BRAESS_NUM_PLAYER,
        "A->B": BRAESS_NUM_PLAYER,
        "A->C": BRAESS_NUM_PLAYER,
        "B->C": BRAESS_NUM_PLAYER,
        "B->D": BRAESS_NUM_PLAYER,
        "C->D": BRAESS_NUM_PLAYER,
        "D->E": BRAESS_NUM_PLAYER
    },
    free_flow_travel_time={
        "O->A": 0,
        "A->B": 1.0,
        "A->C": 2.0,
        "B->C": 0.25,
        "B->D": 2.0,
        "C->D": 1.0,
        "D->E": 0
    })

BRAESS_NETWORK_VEHICLES_DEMAND = [
    dynamic_routing_utils.Vehicle("O->A", "D->E")
    for _ in range(BRAESS_NUM_PLAYER)
]

BRAESS_NETWORK_OD_DEMAND = [
    dynamic_routing_utils.OriginDestinationDemand("O->A", "D->E", 0,
                                                  BRAESS_NUM_PLAYER)
]
