use std::{
    env::var, error::Error, fs::read_dir, os::unix::ffi::OsStrExt, path::PathBuf, process::Command,
};

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    let out_dir: String = var("OUT_DIR").unwrap();
    let hipcc = var("HIPCC");
    let hipcc = hipcc.as_deref().unwrap_or("hipcc");
    let hip_paths = get_hip_paths();
    let mut obj_files = Vec::new();
    let mut children = Vec::new();
    for path in hip_paths {
        let display = path.display().to_string();
        println!("cargo:rerun-if-changed={display}");
        let file_name = path.file_name().unwrap();
        let obj_file = PathBuf::from(&out_dir).join(file_name).with_extension("o");
        let child = Command::new(hipcc)
            .arg("-c")
            .arg(&path)
            .arg("-o")
            .arg(&obj_file)
            .args(["-O3", "-fPIC", "--offload-arch=native", "-ffast-math"])
            .spawn()?;
        children.push((display, child));
        obj_files.push(obj_file);
    }
    for (src_path, mut child) in children {
        let status = child.wait()?;
        if !status.success() {
            return Err(format!("hipcc compilation failed for: {}", src_path).into());
        }
    }
    if !Command::new("ar")
        .arg("rcs")
        .arg(PathBuf::from(&out_dir).join("libkernels.a"))
        .args(&obj_files)
        .status()?
        .success()
    {
        return Err("ar command failed to create archive.".into());
    }
    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=kernels");
    link_rocm_hip_lib();
    Ok(())
}

fn get_hip_paths() -> Vec<PathBuf> {
    read_dir("kernels")
        .unwrap()
        .filter_map(|entry| {
            let path = entry.unwrap().path();
            if path.extension().map(|e| e.as_bytes()) == Some(b"hip") {
                Some(path)
            } else {
                None
            }
        })
        .collect()
}

fn link_rocm_hip_lib() {
    let rocm_path = var("HIP_PATH")
        .or_else(|_| var("ROCM_PATH"))
        .unwrap_or_else(|_| String::from("/opt/rocm"));
    println!("cargo:rustc-link-search=native={rocm_path}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}
