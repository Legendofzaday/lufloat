use std::env::var;
use std::error::Error;
use std::ffi::OsStr;
use std::fs::read_dir;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=kernels");
    let out_dir = var("OUT_DIR")?;
    let hip_paths = read_dir("kernels")?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension() == Some(OsStr::new("hip")))
        .collect::<Vec<PathBuf>>();
    let mut obj_files = Vec::new();
    let mut children = Vec::new();
    for path in hip_paths {
        println!("cargo:rerun-if-changed={}", path.display());
        let file_stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("Path has no valid UTF-8 stem: {}", path.display()))?;
        let obj_file = format!("{}/{}.o", out_dir, file_stem);
        let path_str = path
            .to_str()
            .ok_or_else(|| format!("Path contains invalid UTF-8: {}", path.display()))?;
        let child = Command::new("hipcc")
            .args(["-c", path_str, "-o", &obj_file, "-O3", "-fPIC"])
            .spawn()?;
        children.push((path_str.to_string(), child));
        obj_files.push(obj_file);
    }
    for (src_path, mut child) in children {
        let status = child.wait()?;
        if !status.success() {
            return Err(format!("hipcc compilation failed for: {}", src_path).into());
        }
    }
    let mut ar_cmd = Command::new("ar");
    ar_cmd.arg("rcs").arg(format!("{}/libkernels.a", out_dir));
    for obj in &obj_files {
        ar_cmd.arg(obj);
    }
    let ar_status = ar_cmd.status()?;
    if !ar_status.success() {
        return Err(format!("ar command failed with status: {}", ar_status).into());
    }
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=kernels");
    let rocm_lib = var("ROCM_PATH")
        .map(|path| format!("{}/lib", path))
        .unwrap_or_else(|_| "/opt/rocm/lib".to_string());
    println!("cargo:rustc-link-search=native={}", rocm_lib);
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    Ok(())
}
