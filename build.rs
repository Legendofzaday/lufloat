use std::{
    env::var,
    fs::read_dir,
    path::PathBuf,
    process::{Child, Command},
};

fn main() {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    let out_dir = var("OUT_DIR").unwrap();
    let collected = compile_lib(&out_dir);
    let mut obj_files = Vec::with_capacity(collected.len());
    for (mut child, obj_file) in collected {
        assert!(child.wait().unwrap().success());
        obj_files.push(obj_file);
    }
    static_archive_lib(&out_dir, &obj_files);
    link_lib(&out_dir);
}

fn compile_lib(out_dir: &str) -> Vec<(Child, PathBuf)> {
    read_dir("kernels")
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            let display = path.display();
            println!("cargo:rerun-if-changed={display}");
            let name = path.file_name().unwrap();
            let obj = PathBuf::from(out_dir).join(name).with_extension("o");
            let child = Command::new("hipcc")
                .arg("-c")
                .arg(&path)
                .arg("-o")
                .arg(&obj)
                .args([
                    "-Ofast",
                    "-fPIC",
                    "-fgpu-flush-denormals-to-zero",
                    "--gpu-max-threads-per-block=256",
                    "-munsafe-fp-atomics",
                    "--offload-arch=native",
                ])
                .spawn()
                .unwrap();
            (child, obj)
        })
        .collect()
}

fn static_archive_lib(out_dir: &str, obj_files: &[PathBuf]) {
    assert!(
        Command::new("ar")
            .arg("rcs")
            .arg(PathBuf::from(out_dir).join("libkernels.a"))
            .args(obj_files)
            .status()
            .unwrap()
            .success()
    );
}

fn link_lib(out_dir: &str) {
    let rocm_hip_path = var("HIP_PATH")
        .or_else(|_| var("ROCM_PATH"))
        .unwrap_or_else(|_| String::from("/opt/rocm"));
    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=kernels");
    println!("cargo:rustc-link-search=native={rocm_hip_path}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}
