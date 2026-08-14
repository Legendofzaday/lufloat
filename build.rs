use std::{
    env::{VarError, var},
    error::Error,
    ffi::OsStr,
    fs::{DirEntry, read_dir},
    io,
    path::PathBuf,
    process::{Child, Command, ExitStatus},
};

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIPCC");
    let out_dir: String = var("OUT_DIR")?;
    let hip_paths: Vec<PathBuf> = read_dir("kernels")?
        .map(|entry_result: Result<DirEntry, io::Error>| {
            entry_result.map(|entry: DirEntry| entry.path())
        })
        .filter(
            |path_result: &Result<PathBuf, io::Error>| match path_result {
                Ok(path) => path.is_file() && path.extension() == Some(OsStr::new("hip")),
                Err(_) => true,
            },
        )
        .collect::<Result<Vec<PathBuf>, io::Error>>()?;
    let mut obj_files: Vec<PathBuf> = Vec::new();
    let mut children: Vec<(String, Child)> = Vec::new();
    for path in hip_paths {
        println!("cargo:rerun-if-changed={}", path.display());
        let file_stem: &str = path
            .file_stem()
            .and_then(|s: &OsStr| s.to_str())
            .ok_or_else(|| format!("Path has no valid UTF-8 stem: {}", path.display()))?;
        let obj_file: PathBuf = PathBuf::from(&out_dir).join(format!("{file_stem}.o"));
        let path_str: &str = path
            .to_str()
            .ok_or_else(|| format!("Path contains invalid UTF-8: {}", path.display()))?;
        let compiler: String = var("HIPCC").unwrap_or_else(|_err: VarError| "hipcc".to_string());
        let child: Child = Command::new(compiler)
            .args(["-c", path_str, "-o"])
            .arg(&obj_file)
            .args(["-O3", "-fPIC", "--offload-arch=native", "-ffast-math"])
            .spawn()?;
        children.push((path_str.to_string(), child));
        obj_files.push(obj_file);
    }
    for (src_path, mut child) in children {
        let status: ExitStatus = child.wait()?;
        if !status.success() {
            return Err(format!("hipcc compilation failed for: {}", src_path).into());
        }
    }
    if !obj_files.is_empty() {
        let mut ar_cmd: Command = Command::new("ar");
        ar_cmd
            .arg("rcs")
            .arg(PathBuf::from(&out_dir).join("libkernels.a"));

        for obj in &obj_files {
            ar_cmd.arg(obj);
        }
        let ar_status: ExitStatus = ar_cmd.status()?;
        if !ar_status.success() {
            return Err(format!("ar command failed with status: {}", ar_status).into());
        }
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=static=kernels");
    }
    let rocm_lib: String = var("ROCM_PATH")
        .map(|path: String| format!("{}/lib", path))
        .unwrap_or_else(|_err: VarError| "/opt/rocm/lib".to_string());
    println!("cargo:rustc-link-search=native={}", rocm_lib);
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    Ok(())
}
