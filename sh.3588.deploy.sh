PATH_REMOTE=root@10.42.0.161
# PATH_REMOTE=root@192.168.2.250
PATH_SO=$PATH_REMOTE:/usr/lib/aarch64-linux-gnu
PATH_DEPLOY=$PATH_REMOTE:/root/rknn-rs
PATH_BINARY=target/aarch64-unknown-linux-gnu/release

export PKG_CONFIG_ALLOW_CROSS=1
export PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
cargo build --target aarch64-unknown-linux-gnu \
            --release \
            --example bench

scp ./c_lib/librknnrt.so $PATH_SO 
scp -r ./models/* \
    ../images/cig_2026_01_21 \
    $PATH_BINARY/examples/bench \
    $PATH_DEPLOY