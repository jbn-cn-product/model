extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let c_lib_path = PathBuf::from(manifest_dir).join("c_lib");
    
    println!("cargo:rustc-link-search=native={}", c_lib_path.display());
    println!("cargo:rustc-link-lib=dylib=rknnrt");
    println!("cargo:rerun-if-changed=c_lib/rknn_api.h");

    // bindgen
    let target = env::var("TARGET").unwrap_or_else(|_| env::var("HOST").unwrap());
    
    let mut builder = bindgen::Builder::default()
        .header("c_lib/rknn_api.h")
        .raw_line("#[allow(dead_code)]")
        .raw_line("#[allow(unused_imports)]");
    
    // Configure for cross-compilation to aarch64
    if target.contains("aarch64") {
        builder = builder
            .clang_arg("--target=aarch64-unknown-linux-gnu")
            .clang_arg("--sysroot=/usr/aarch64-linux-gnu")
            .clang_arg("-I/usr/aarch64-linux-gnu/include")
            .clang_arg("-nostdinc");
    }
    
    let bindings = builder
        .generate()
        .expect("bindgen failed");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("bindgen output none");
}