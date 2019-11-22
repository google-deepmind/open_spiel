game = Spiel.load_game("kuhn_poker")
state = Spiel.new_initial_state(game)

while !Spiel.is_terminal(state)
    if Spiel.is_chance_node(state)
        outcomes_with_probs = Spiel.chance_outcomes(state)
        action = rand([x.first for x in outcomes_with_probs])
        Spiel.apply_action(state, action)
    else
        Spiel.apply_action(state, Spiel.legal_actions(state)[1])
    end
end