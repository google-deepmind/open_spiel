@testset "Games" begin
    # Put a bound on length of game so test does not timeout.
    MAX_ACTIONS_PER_GAME = 1000

    # All games registered in the main spiel library.
    SPIEL_GAMES_LIST = registered_games()

end