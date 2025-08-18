use std::sync::OnceLock;

static TERMINAL: OnceLock<String> = OnceLock::new();

pub fn user_agent() -> String {
    TERMINAL.get_or_init(detect_terminal).to_string()
}

fn detect_terminal() -> String {
    if let Ok(tp) = std::env::var("TERM_PROGRAM") {
        if !tp.trim().is_empty() {
            let ver = std::env::var("TERM_PROGRAM_VERSION").ok();
            return match ver {
                Some(v) if !v.trim().is_empty() => format!("{tp}/{v}"),
                _ => tp,
            };
        }
    }

    if let Ok(v) = std::env::var("WEZTERM_VERSION") {
        if !v.trim().is_empty() {
            return format!("WezTerm/{v}");
        }
        return "WezTerm".to_string();
    }

    if std::env::var("KITTY_WINDOW_ID").is_ok()
        || std::env::var("TERM")
            .map(|t| t.contains("kitty"))
            .unwrap_or(false)
    {
        return "kitty".to_string();
    }

    if std::env::var("ALACRITTY_SOCKET").is_ok()
        || std::env::var("TERM")
            .map(|t| t == "alacritty")
            .unwrap_or(false)
    {
        return "Alacritty".to_string();
    }

    if let Ok(v) = std::env::var("KONSOLE_VERSION") {
        if !v.trim().is_empty() {
            return format!("Konsole/{v}");
        }
        return "Konsole".to_string();
    }

    if std::env::var("GNOME_TERMINAL_SCREEN").is_ok() {
        return "gnome-terminal".to_string();
    }
    if let Ok(v) = std::env::var("VTE_VERSION") {
        if !v.trim().is_empty() {
            return format!("VTE/{v}");
        }
        return "VTE".to_string();
    }

    if std::env::var("WT_SESSION").is_ok() {
        return "WindowsTerminal".to_string();
    }

    std::env::var("TERM").unwrap_or_else(|_| "unknown".to_string())
}
