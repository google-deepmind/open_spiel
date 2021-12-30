extern crate libc;

use std::os::raw::c_void;
use libc::c_char;
use libc::c_double;
// use libc::c_int;
use libc::c_long;
use libc::free;
use std::ffi::CStr;
use std::str;
use std::slice;

include!("./open_spiel_bindings.rs");

pub struct State {
  state: *mut c_void,
}

pub struct Game {
  game: *mut c_void,
}

impl State {
  pub fn new(sptr: *mut c_void) -> State {
    State {
      state: sptr
    }
  }

  pub fn current_player(&self) -> i32 {
    return (unsafe { StateCurrentPlayer(self.state) }) as i32;
  }

  pub fn clone(&self) -> State {
    return unsafe { State { state: StateClone(self.state) } };
  }

  pub fn is_chance_node(&self) -> bool {
    let ret = unsafe { StateIsChanceNode(self.state) };
    if ret == 0 {
      return false;
    } else {
      return true;
    }
  }
  
  pub fn is_terminal(&self) -> bool {
    let ret = unsafe { StateIsTerminal(self.state) };
    if ret == 0 {
      return false;
    } else {
      return true;
    }
  }

  pub fn num_players(&self) -> usize {
    return (unsafe { StateNumPlayers(self.state) }) as usize;
  }

  pub fn returns(&self) -> Vec<f64> {
    let length = self.num_players();
    let c_buf: *mut c_double = unsafe { StateReturns(self.state) };
    let mut returns_vec = vec![0.0 as f64; length];
    unsafe {
      let slice = slice::from_raw_parts(c_buf, length);
      for i in 0..length {
        returns_vec[i] = slice[i];
      }
      free(c_buf as *mut c_void);
    }
    return returns_vec;
  }

  pub fn player_return(&self, player: i32) -> f64 {
    let val: f64 = unsafe { StatePlayerReturn(self.state, player) };
    return val;
  }

  pub fn legal_actions(&self) -> Vec<i64> {
    let mut c_num_legal_moves = 0;
    let c_buf: *mut c_long = unsafe {
        StateLegalActions(self.state, &mut c_num_legal_moves)
    };
    let length: usize = c_num_legal_moves as usize;
    let mut legal_actions_vec = vec![0; length];
    unsafe {
      let slice = slice::from_raw_parts(c_buf, length);
      for i in 0..length {
        legal_actions_vec[i] = slice[i];
      }
      free(c_buf as *mut c_void);
    }
    return legal_actions_vec;
  }

  pub fn apply_action(&self, action: i64) {
    unsafe { StateApplyAction(self.state, action) }
  }

  pub fn action_to_string(&self, player: i32, action: i64) -> String {
    let c_buf: *mut c_char = unsafe { 
      StateActionToString(self.state, player, action)
    };
    let c_str: &CStr = unsafe { CStr::from_ptr(c_buf) };
    let str_slice: &str = c_str.to_str().unwrap();
    let str_buf: String = str_slice.to_owned();
    unsafe { free(c_buf as *mut c_void) };
    return str_buf;
  }
  
  pub fn to_string(&self) -> String {
    let c_buf: *mut c_char = unsafe { StateToString(self.state) };
    let c_str: &CStr = unsafe { CStr::from_ptr(c_buf) };
    let str_slice: &str = c_str.to_str().unwrap();
    let str_buf: String = str_slice.to_owned();
    unsafe { free(c_buf as *mut c_void) };
    return str_buf;
  }
}

impl Drop for State {
  fn drop(&mut self) {
     unsafe { DeleteState(self.state) }
  }
}

impl Game {
  pub fn new(game_name: String) -> Game {
    Game {
      game: unsafe { LoadGame(game_name.as_ptr() as *const i8) }
    }
  }
  
  pub fn short_name(&self) -> String {
    let c_buf: *mut c_char = unsafe { GameShortName(self.game) };
    let c_str: &CStr = unsafe { CStr::from_ptr(c_buf) };
    let str_slice: &str = c_str.to_str().unwrap();
    let str_buf: String = str_slice.to_owned();
    unsafe { free(c_buf as *mut c_void) };
    return str_buf;
  }
  
  pub fn long_name(&self) -> String {
    let c_buf: *mut c_char = unsafe { GameLongName(self.game) };
    let c_str: &CStr = unsafe { CStr::from_ptr(c_buf) };
    let str_slice: &str = c_str.to_str().unwrap();
    let str_buf: String = str_slice.to_owned();
    unsafe { free(c_buf as *mut c_void) };
    return str_buf;
  }

  pub fn new_initial_state(&self) -> State {
    return unsafe { State::new(GameNewInitialState(self.game)) };
  }

  pub fn num_players(&self) -> i32 {
    return (unsafe { GameNumPlayers(self.game) }) as i32;
  }
  
  pub fn max_game_length(&self) -> i32 {
    return (unsafe { GameMaxGameLength(self.game) }) as i32;
  }
  
  pub fn num_distinct_actions(&self) -> i32 {
    return (unsafe { GameNumDistinctActions(self.game) }) as i32;
  }
}

impl Drop for Game {
  fn drop(&mut self) {
     unsafe { DeleteGame(self.game) }
  }
}

