use std::{env, fs, path, process};

fn main() {
    println!("cargo:rerun-if-changed=hip");
    let out_dir: String = unsafe { env::var("OUT_DIR").unwrap_unchecked() };
    let lib_file: String = format!("{}/lib{}.a", out_dir, "hip");
    let mut obj_files: Vec<String> = Vec::new();
    let mut childs: Vec<process::Child> = Vec::new();
    let entries: fs::ReadDir = unsafe { fs::read_dir("hip").unwrap_unchecked() };
    for entry in entries {
        let path: path::PathBuf = unsafe { entry.unwrap_unchecked().path() };
        println!("cargo:rerun-if-changed={}", path.display());
        let file_stem: &str = unsafe {
            path.file_stem()
                .unwrap_unchecked()
                .to_str()
                .unwrap_unchecked()
        };
        let obj_file: String = format!("{}/{}.o", out_dir, file_stem);
        let child: process::Child = unsafe {
            process::Command::new("hipcc")
                .args([
                    "-c",
                    path.to_str().unwrap_unchecked(),
                    "-o",
                    &obj_file,
                    "-O3",
                    "-fPIC",
                ])
                .spawn()
                .unwrap_unchecked()
        };
        childs.push(child);
        obj_files.push(obj_file);
    }
    for mut child in childs {
        unsafe { child.wait().unwrap_unchecked() };
    }
    let mut ar_cmd = process::Command::new("ar");
    ar_cmd.arg("rcs").arg(&lib_file);
    for obj in &obj_files {
        ar_cmd.arg(obj);
    }
    unsafe { ar_cmd.status().unwrap_unchecked() };
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=hip");
    if path::Path::new("/usr/lib64/libamdhip64.so").exists() {
        println!("cargo:rustc-link-search=native=/usr/lib64");
    } else {
        println!("cargo:rustc-link-search=native=/opt/rocm/lib");
    }
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}
