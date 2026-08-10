use std::env::var;
use std::error::Error;
use std::ffi::OsStr;
use std::fs::read_dir;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIPCC");
    let out_dir = var("OUT_DIR")?;
    let hip_paths = read_dir("kernels")?
        .map(|entry_result| entry_result.map(|entry| entry.path()))
        .filter(|path_result| match path_result {
            Ok(path) => path.is_file() && path.extension() == Some(OsStr::new("hip")),
            Err(_) => true,
        })
        .collect::<Result<Vec<PathBuf>, std::io::Error>>()?;
    let mut obj_files = Vec::new();
    let mut children = Vec::new();
    for path in hip_paths {
        println!("cargo:rerun-if-changed={}", path.display());
        let file_stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("Path has no valid UTF-8 stem: {}", path.display()))?;
        let obj_file = PathBuf::from(&out_dir).join(format!("{file_stem}.o"));
        let path_str = path
            .to_str()
            .ok_or_else(|| format!("Path contains invalid UTF-8: {}", path.display()))?;
        let compiler = var("HIPCC").unwrap_or_else(|_| "hipcc".to_string());
        let child = Command::new(compiler)
            .args(["-c", path_str, "-o"])
            .arg(&obj_file)
            .args(["-O3", "-fPIC", "--offload-arch=native", "-ffast-math"])
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
    if !obj_files.is_empty() {
        let mut ar_cmd = Command::new("ar");
        ar_cmd
            .arg("rcs")
            .arg(PathBuf::from(&out_dir).join("libkernels.a"));
        for obj in &obj_files {
            ar_cmd.arg(obj);
        }
        let ar_status = ar_cmd.status()?;
        if !ar_status.success() {
            return Err(format!("ar command failed with status: {}", ar_status).into());
        }
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=static=kernels");
    }
    let rocm_lib = var("ROCM_PATH")
        .map(|path| format!("{}/lib", path))
        .unwrap_or_else(|_| "/opt/rocm/lib".to_string());
    println!("cargo:rustc-link-search=native={}", rocm_lib);
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    Ok(())
}
