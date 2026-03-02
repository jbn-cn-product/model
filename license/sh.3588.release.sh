#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PKG_CONFIG_ALLOW_CROSS=1
export PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH

export ORT_STRATEGY=system
export ORT_LIB_LOCATION="$SCRIPT_DIR/c_lib/libonnxruntime.so"
export RUSTFLAGS="-L native=$SCRIPT_DIR/c_lib"

RUST_BACKTRACE=1 cargo build --release --target aarch64-unknown-linux-gnu --example license --example cig