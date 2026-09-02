use std::sync::Mutex;

struct BackendState {
    #[allow(dead_code)]
    sidecar_pid: Mutex<Option<u32>>,
}

#[tauri::command]
fn get_backend_url() -> String {
    "http://127.0.0.1:8766".to_string()
}

#[tauri::command]
fn start_backend() -> Result<String, String> {
    // In development mode, the Python backend is started manually
    // In production, this would launch the bundled sidecar
    Ok("http://127.0.0.1:8766".to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(BackendState {
            sidecar_pid: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![get_backend_url, start_backend])
        .run(tauri::generate_context!())
        .expect("error while running aura studio");
}
