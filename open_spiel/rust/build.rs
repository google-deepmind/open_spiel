use std::env;

fn main() {
  let cwd = env::current_dir().unwrap();
  let path_str = cwd.into_os_string().into_string().unwrap();
  println!("cargo:rustc-link-search={}", path_str);
  println!("cargo:rustc-link-lib=dylib=rust_spiel");
}
