fn main() {
    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-env=ORT_DYLIB_PATH=C:\\Users\\Limited\\onnxruntime-win-x64-1.22.0\\onnxruntime-win-x64-1.22.0\\lib\\onnxruntime.dll");
        println!("cargo:rustc-env=ORT_SKIP_DOWNLOAD=1");
        println!("cargo:rustc-env=ORT_LIB_LOCATION=C:\\Users\\Limited\\onnxruntime-win-x64-1.22.0\\onnxruntime-win-x64-1.22.0\\lib");
        // println!("cargo:rustc-link-search=native=E:\\faiss-1.13.1\\faiss-1.13.1\\build_noBLAS_noLAPACK\\c_api\\Release");
        // println!("cargo:rustc-link-lib=faiss_c");
    }
}
