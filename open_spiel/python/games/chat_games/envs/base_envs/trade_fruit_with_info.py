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

"""A base environment for trading fruit with private info.
"""

import dataclasses

from open_spiel.python.games.chat_games.envs.comm_substrates import trades
from open_spiel.python.games.chat_games.envs.scenarios.domains import trade_fruit
from open_spiel.python.games.chat_games.envs.utils import header
from open_spiel.python.games.chat_games.envs.utils import text


action_keys = tuple([])
info_keys = tuple(['fruit_endowment', 'fruit_valuations'])

w_opts = (trades.W_OPTS_PREFIX +
          'Fruit Endowment:\n{fruit_endowment}\n\n' +
          'Fruit Valuations:\n{fruit_valuations}\n' +
          trades.PLAIN)

# Example a
email_1a = ['Hi Joel,',
            'I would like to trade you 2 strawberries for 3 blueberries.',
            'Would you like to trade with me?',
            'Best,', 'Bob']
email_1a = (trades.PLAIN.format(sender='Alicia', receiver='Joel') +
            '\n\n'.join(text.wrap(email_1a)))

email_2a = ['Hi Alicia,',
            'Thanks for reaching out. I only have 2 blueberries, but even if ' +
            'I had 3, I would not want to give them up. Also, I dislike ' +
            'strawberries. I do not think a trade makes sense in this case.',
            'Thanks for considering trading with me though!',
            'Best,', 'Joel']
email_2a = (trades.PLAIN.format(sender='Joel', receiver='Alicia') +
            '\n\n'.join(text.wrap(email_2a)))

email_3a = ['Hi Joel,',
            'That is all well. I understand.',
            'Have a good day!',
            'Best,', 'Alicia']
email_3a = (trades.PLAIN.format(sender='Alicia', receiver='Joel') +
            '\n\n'.join(text.wrap(email_3a)))

example_a = email_1a + email_2a
example_a = example_a.strip('\n')

# Example b
email_1b = ['Hi Marcus,',
            'I would like to trade you 2 kiwis for 1 watermelon.',
            'Would you like to trade with me?',
            'Best,', 'Taylor']
email_1b = (trades.PLAIN.format(sender='Taylor', receiver='Marcus') +
            '\n\n'.join(text.wrap(email_1b)))

email_2b = ['Hi Taylor,',
            'I love kiwis! And lucky for you, I have a watermelon.',
            'Lets trade!',
            'Best,', 'Marcus']
email_2b = (trades.PLAIN.format(sender='Marcus', receiver='Taylor') +
            '\n\n'.join(text.wrap(email_2b)))

email_3b = ['Hi Marcus,',
            'Great! It was a pleasure negotiating with you.',
            'Have a good day!',
            'Best,', 'Taylor']
email_3b = (trades.PLAIN.format(sender='Taylor', receiver='Marcus') +
            '\n\n'.join(text.wrap(email_3b)))

example_b = email_1b + email_2b + email_3b
example_b = example_b.strip('\n')

# Example c
email_1c = ['Hi Suzy,',
            'I would like to trade you 1 banana for 1 apple.',
            'Would you like to trade with me?',
            'Best,', 'Bob']
email_1c = (trades.PLAIN.format(sender='Bob', receiver='Suzy') +
            '\n\n'.join(text.wrap(email_1c)))

email_2c = ['Hi Bob,',
            'Thanks for reaching out. I really like my apples so I am ' +
            'hesitant to give them up. Would you be willing to take a few ' +
            'kiwis instead? I would like to trade you 3 kiwis for 1 banana.',
            'Does that work?',
            'Best,', 'Suzy']
email_2c = (trades.PLAIN.format(sender='Suzy', receiver='Bob') +
            '\n\n'.join(text.wrap(email_2c)))

email_3c = ['Hi Suzy,',
            'Yes! I would have preferred an apple but 3 kiwis are nearly as ' +
            'good and I would rather have those than a banana.',
            'Thanks for trading with me!',
            'Best,', 'Bob']
email_3c = '\n\n'.join(text.wrap(email_3c))

example_c = email_1c + email_2c
example_c = example_c.strip('\n')

instr_a = ['You are an assistant who is playing a game where you trade fruit.' +
           ' You want to make a trade that is best for you. You will read a ' +
           'dialogue that contains a conversation where you have been ' +
           'negotiating to trade your fruit for another persons fruit. You ' +
           'will then read a text block that contains information a) about ' +
           'the actual fruit you currently have and are able to trade and b)' +
           ' information about how much you value certain types of fruit.',
           'You should use everything you learned from this to decide to ',
           '1) accept the trade if you are happy with the terms,',
           '2) reject the negotiation all together and say goodbye if you do ' +
           'not think an agreement can be reached,',
           '3) counter-propose an alternative trade that includes what fruit ' +
           'you would like to give and what fruit you would like to receive ' +
           'in turn.',
           'Consider the following example dialogues. Components of the ' +
           'examples will be demarked with the symbol "&". Here is the first ' +
           'example which shows a trade is rejected.',
           '&' * 50]
instr_b = ['&' * 50,
           'Here is a second example where a trade is accepted.',
           '&' * 50]
instr_c = ['&' * 50,
           'Here is a partial dialogue where we demonstrate a reasonable ' +
           'countertrade.',
           '&' * 50]
instr_d = ['&' * 50,
           'Continuing the example. You now see the fruit you have and how ' +
           'much you value each fruit type.',
           '&' * 50]
info = w_opts.format(sender='Bob', receiver='Suzy',
                     fruit_endowment=trade_fruit.ENDOWMENT_A,
                     fruit_valuations=trade_fruit.VALUATION_A).strip('\n')
instr_e = ['&' * 50,
           'A reasonable way to respond would be as follows:',
           '&' * 50]
instr_f = ['&' * 50,
           'Now you are going to read a fresh dialogue, fruit endowment, and ' +
           'fruit valuation information. Please give a reasonable response ' +
           'that attempts to reach an agreement to trade fruit.',
           '&' * 50]
context = (text.wrap(instr_a) + [example_a] + text.wrap(instr_b) +[example_b] +
           text.wrap(instr_c) + [example_c] + text.wrap(instr_d) + [info] +
           text.wrap(instr_e) + [email_3c] + text.wrap(instr_f))

HEADER = header.Header(trades.PLAIN,
                       w_opts,
                       trades.strip_msg,
                       trades.SPECIAL_CHARS,
                       action_keys,
                       info_keys,
                       '\n\n'.join(context))


@dataclasses.dataclass(frozen=True)
class Scenario(header.BaseScenario):
  fruit_endowment: str
  fruit_valuations: str
