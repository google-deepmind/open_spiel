# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Examples of negotiation scenarios -- useful for generating more examples.
"""

from open_spiel.python.games.chat_games.envs.utils import text

# negotiating rent (money)
SCENARIO_A_LIST = [
    'Hi {receiver},', 'I hope you are well,', 'I understand you have been a ' +
    'long time tenant with me, so I hate to increase rent, but as you know ' +
    'inflation has increased by 6 percent recently. In order to stay ' +
    'solvent I will need to increase your rent by 6 percent as well. I hope ' +
    'you understand my thinking.\n\nHow do you feel about this? Would you ' +
    'like to continue renting from me?', 'Best,', '{sender}']
SCENARIO_A = '\n\n'.join(text.wrap(SCENARIO_A_LIST))

# negotiating deadline extension (time)
SCENARIO_B_LIST = [
    'Dear {receiver},', 'I understand that my payment is due at the end of ' +
    'this month, but I will find it hard to come up with the money. Would it ' +
    'be possible to extend the due date by 1 week? This would allow me to ' +
    'come up with the necessary funds. As a concession, I would be willing to' +
    ' pay early next month.', 'How do you feel about this? Do you have any ' +
    'other alternatives that you would be happy with?', 'Best,', '{sender}']
SCENARIO_B = '\n\n'.join(text.wrap(SCENARIO_B_LIST))

# negotiating a trade (things)
SCENARIO_C_LIST = [
    'Hey {receiver},', 'Thanks for your interest in my baseball card  ' +
    'collection. I see you like my signed Babe Ruth special edition card. To ' +
    'be honest, I really like your signed Nolan Ryan jersey. I also like ' +
    'your signed Roger Clemens ball. Would you be interested in a trade? I ' +
    'have a few other things you might like to sweeten the deal: Ken Griffey '+
    'Jr baseball bat, Mike Trout signed card, ...', 'What do you think?',
    'Best,', '{sender}']
SCENARIO_C = '\n\n'.join(text.wrap(SCENARIO_C_LIST))
