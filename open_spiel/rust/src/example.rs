use rust_open_spiel::*;

fn main() {
  let game: Game = Game::new(String::from("tic_tac_toe"));
  println!("The short name is: {}", game.short_name());
  println!("The long name is: {}", game.long_name());
  println!("Number of players: {}", game.num_players());
  println!("Number of distinct actions: {}", game.num_distinct_actions());
  println!("Max game length: {}", game.max_game_length());

  let state: State = game.new_initial_state();
  println!("Initial state:\n{}", state.to_string());

  let clone: State = state.clone();
  println!("Cloned initial state:\n{}", clone.to_string());

  while !state.is_terminal() {
    println!("");
    println!("State:\n{}", state.to_string());
    let legal_actions: Vec<i64> = state.legal_actions();
    let player: i32 = state.current_player();
    println!("Legal actions: ");
    let action = legal_actions[0];
    for a in legal_actions {
      println!("  {}: {}", a, state.action_to_string(player, a));
    }
    println!("Taking action {}: {}", action,
                                     state.action_to_string(player, action));
    state.apply_action(action);
  }

  println!("Terminal state reached:\n{}\n", state.to_string());
  let returns: Vec<f64> = state.returns();
  for i in 0..game.num_players() {
    println!("Utility for player {} is {}", i, returns[i as usize]);
  }
}

